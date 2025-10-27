import re
import csv
import yaml
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
import requests

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate


@dataclass
class TranscriptLine:
    """Строка транскрипции"""
    start_time: float
    end_time: float
    text: str


@dataclass
class Topic:
    """Тема с временными рамками"""
    number: int
    start_time: str
    end_time: str
    summary: str
    full_text: str


class Config:
    """Управление конфигурацией"""

    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Конфиг не найден, используются настройки по умолчанию")
            self.config = self._default_config()

    def _default_config(self):
        return {
            'model': {
                'name': 'mistral',
                'temperature': 0.1,
                'num_ctx': 8192,
                'timeout': 30,
                'base_url': 'http://localhost:11434'
            },
            'processing': {
                'chunk_lines': 100,
                'min_topic_duration': 30,
                'max_topic_duration': 300,
                'pause_threshold': 10,
                'use_llm': True,
                'debug': True
            },
            'paths': {
                'input_dir': 'transcripts',
                'output_dir': 'results'
            }
        }

    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value


class TranscriptAnalyzer:
    """Анализатор транскрипций"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config(config_path)
        self.debug = self.config.get('processing', 'debug', default=True)

        # Параметры обработки
        self.chunk_lines = self.config.get('processing', 'chunk_lines', default=100)
        self.min_duration = self.config.get('processing', 'min_topic_duration', default=30)
        self.max_duration = self.config.get('processing', 'max_topic_duration', default=300)
        self.pause_threshold = self.config.get('processing', 'pause_threshold', default=10)
        self.use_llm = self.config.get('processing', 'use_llm', default=True)
        self.llm_timeout = self.config.get('model', 'timeout', default=30)
        self.base_url = self.config.get('model', 'base_url', default='http://localhost:11434')

        # Проверка доступности Ollama
        self._check_ollama_connection()

        # Инициализация LLM
        self.llm = None
        if self.use_llm:
            self._init_llm()

        self._init_prompts()

    def _check_ollama_connection(self):
        """Проверка подключения к Ollama"""
        print(f"🔍 Проверка подключения к Ollama ({self.base_url})...")

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"  ✓ Ollama доступен, найдено моделей: {len(models)}")

                model_name = self.config.get('model', 'name', default='mistral')
                model_found = any(m['name'].startswith(model_name) for m in models)

                if model_found:
                    print(f"  ✓ Модель '{model_name}' найдена")
                else:
                    print(f"  ⚠️  Модель '{model_name}' не найдена!")
                    print(f"  Доступные модели: {[m['name'] for m in models]}")

                return True
            else:
                print(f"  ✗ Ollama вернул код {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"  ✗ Не удалось подключиться к Ollama!")
            print(f"  Убедитесь, что Ollama запущен: ollama serve")
            return False
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            return False

    def _init_llm(self):
        """Инициализация LLM"""
        try:
            model_name = self.config.get('model', 'name', default='mistral')
            print(f"🤖 Инициализация LLM (модель: {model_name})...")

            self.llm = OllamaLLM(
                model=model_name,
                base_url=self.base_url,
                temperature=self.config.get('model', 'temperature', default=0.1),
                num_ctx=self.config.get('model', 'num_ctx', default=8192),
            )

            # Тестовый запрос
            print("  Тестовый запрос к LLM...", end=' ')
            start = time.time()
            test_response = self.llm.invoke("Привет, ответь одним словом")
            elapsed = time.time() - start
            print(f"✓ ({elapsed:.2f}s)")

            if self.debug:
                print(f"  Ответ LLM: {test_response[:100]}")

        except Exception as e:
            print(f"  ✗ Ошибка инициализации LLM: {e}")
            self.llm = None
            self.use_llm = False

    def _init_prompts(self):
        """Инициализация промптов"""

        self.segmentation_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Раздели транскрипцию на смысловые темы.

ТРАНСКРИПЦИЯ:
{text}

Верни JSON массив с темами. Для каждой укажи начало и конец в секундах:
[{{"start": секунды, "end": секунды}}]

ТОЛЬКО JSON массив:"""
        )

        self.summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Кратко опиши тему одним предложением (15-25 слов).

ТЕКСТ:
{text}

КРАТКОЕ ОПИСАНИЕ:"""
        )

    def parse_transcript(self, file_path: Path) -> List[TranscriptLine]:
        """Парсинг файла транскрипции"""
        print(f"📄 Парсинг файла...")
        lines = []

        patterns = [
            r'\[.*?\]\s*(\d+\.?\d*)-(\d+\.?\d*):\s*(.*)',
            r'(\d+\.?\d*)-(\d+\.?\d*):\s*(.*)',
            r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s*(.*)',
        ]

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parsed = False
                for pattern in patterns:
                    match = re.match(pattern, line)
                    if match:
                        groups = match.groups()
                        lines.append(TranscriptLine(
                            start_time=float(groups[0]),
                            end_time=float(groups[1]),
                            text=groups[2].strip()
                        ))
                        parsed = True
                        break

                if not parsed and self.debug and line_num <= 3:
                    print(f"  ⚠️  Не удалось распарсить строку {line_num}: {line[:50]}")

        if self.debug and lines:
            print(f"  Первая строка: {lines[0].start_time}-{lines[0].end_time}: {lines[0].text[:50]}")
            print(f"  Последняя строка: {lines[-1].start_time}-{lines[-1].end_time}: {lines[-1].text[:50]}")

        return lines

    def segment_by_rules(self, lines: List[TranscriptLine]) -> List[Dict]:
        """Разбиение на темы по эвристическим правилам"""
        if self.debug:
            print(f"  📐 Применение эвристических правил...")

        segments = []
        current_segment = []

        for i, line in enumerate(lines):
            current_segment.append(line)

            should_break = False

            if i < len(lines) - 1:
                next_line = lines[i + 1]
                pause = next_line.start_time - line.end_time

                if pause > self.pause_threshold:
                    should_break = True
                    if self.debug:
                        print(f"    Большая пауза: {pause:.1f}s")

                segment_duration = line.end_time - current_segment[0].start_time
                if segment_duration > self.max_duration:
                    should_break = True
                    if self.debug:
                        print(f"    Макс. длительность: {segment_duration:.1f}s")
            else:
                should_break = True

            if should_break and current_segment:
                duration = current_segment[-1].end_time - current_segment[0].start_time

                if duration >= self.min_duration:
                    text = ' '.join([l.text for l in current_segment])
                    segments.append({
                        'start': current_segment[0].start_time,
                        'end': current_segment[-1].end_time,
                        'text': text
                    })
                    if self.debug:
                        print(f"    ✓ Сегмент {len(segments)}: {duration:.1f}s, {len(current_segment)} строк")

                current_segment = []

        return segments

    def segment_with_llm(self, lines: List[TranscriptLine]) -> List[Dict]:
        """Разбиение на темы с помощью LLM"""
        if not self.llm:
            print("  ⚠️  LLM не инициализирован")
            return None

        chunks = []
        for i in range(0, len(lines), self.chunk_lines):
            chunk = lines[i:i + self.chunk_lines]
            chunks.append(chunk)

        print(f"  📦 Разбито на {len(chunks)} чанков по ~{self.chunk_lines} строк")

        all_segments = []

        for chunk_idx, chunk in enumerate(chunks):
            print(f"  🔄 Чанк {chunk_idx + 1}/{len(chunks)}:", end=' ')

            # Сначала правила
            chunk_segments = self.segment_by_rules(chunk)

            if not chunk_segments:
                print("пусто")
                continue

            print(f"правила дали {len(chunk_segments)} сегментов", end=', ')

            # Пробуем LLM
            try:
                sample_lines = chunk[:50]
                text_lines = [
                    f"{line.start_time}-{line.end_time}: {line.text}"
                    for line in sample_lines
                ]
                chunk_text = '\n'.join(text_lines)

                if self.debug:
                    print(f"\n    📝 Текст для LLM ({len(chunk_text)} символов):")
                    print(f"    {chunk_text[:200]}...")

                prompt = self.segmentation_prompt.format(text=chunk_text)

                print("отправка в LLM...", end=' ')
                start_time = time.time()

                response = self.llm.invoke(prompt)

                elapsed = time.time() - start_time
                print(f"получен ответ за {elapsed:.1f}s", end=', ')

                if self.debug:
                    print(f"\n    🤖 Ответ LLM ({len(response)} символов):")
                    print(f"    {response[:300]}...")

                llm_segments = self._parse_json_response(response)

                if llm_segments and len(llm_segments) > 0:
                    print(f"LLM нашел {len(llm_segments)} тем ✓")

                    for llm_seg in llm_segments:
                        start = llm_seg.get('start', 0)
                        end = llm_seg.get('end', 0)

                        matching_lines = [
                            l for l in chunk
                            if l.start_time >= start and l.end_time <= end
                        ]

                        if matching_lines:
                            all_segments.append({
                                'start': matching_lines[0].start_time,
                                'end': matching_lines[-1].end_time,
                                'text': ' '.join([l.text for l in matching_lines])
                            })
                else:
                    print("не удалось распарсить, используем правила")
                    all_segments.extend(chunk_segments)

            except Exception as e:
                print(f"ошибка: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                all_segments.extend(chunk_segments)

        return all_segments if all_segments else None

    def generate_summaries(self, topics: List[Dict]) -> List[Dict]:
        """Генерация саммари для тем"""
        print(f"  💭 Генерация саммари для {len(topics)} тем...")

        for i, topic in enumerate(topics):
            if (i + 1) % 5 == 0 or self.debug:
                print(f"    {i + 1}/{len(topics)}...", end=' ')

            text = topic['text'][:800]

            if self.llm:
                try:
                    if self.debug:
                        print(f"LLM запрос...", end=' ')

                    start = time.time()
                    prompt = self.summary_prompt.format(text=text)
                    summary = self.llm.invoke(prompt).strip()
                    elapsed = time.time() - start

                    if self.debug:
                        print(f"✓ ({elapsed:.1f}s)")

                    if summary and len(summary) > 10:
                        if len(summary) > 200:
                            summary = summary[:197] + "..."
                        topic['summary'] = summary
                        continue
                except Exception as e:
                    if self.debug:
                        print(f"ошибка: {e}")

            topic['summary'] = self._generate_fallback_summary(text)
            if self.debug:
                print("fallback")

        return topics

    def _generate_fallback_summary(self, text: str) -> str:
        """Простое саммари без LLM"""
        text_lower = text.lower()

        keywords = []
        if any(word in text_lower for word in ['вопрос', '?']):
            keywords.append('вопрос-ответ')
        if any(word in text_lower for word in ['обсуждение', 'говорим', 'тема']):
            keywords.append('обсуждение')
        if any(word in text_lower for word in ['проблема', 'ситуация', 'кризис']):
            keywords.append('проблемная ситуация')

        if keywords:
            summary = "Тема: " + ", ".join(keywords)
        else:
            words = [w for w in text.split() if len(w) > 3][:10]
            summary = "Обсуждение: " + " ".join(words)

        return summary[:150]

    def analyze(self, lines: List[TranscriptLine]) -> List[Topic]:
        """Основной метод анализа"""
        if not lines:
            return []

        print(f"  📊 Найдено строк: {len(lines)}")
        total_duration = lines[-1].end_time - lines[0].start_time
        print(f"  ⏱️  Общая длительность: {total_duration/60:.1f} минут")

        print("\n  🔍 Этап 1: Сегментация")
        segments = None

        if self.use_llm and self.llm:
            try:
                segments = self.segment_with_llm(lines)
            except Exception as e:
                print(f"  ✗ Ошибка LLM: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        if not segments:
            print("  📐 Fallback: используем только правила")
            segments = self.segment_by_rules(lines)

        if not segments:
            print("  ✗ Не удалось выделить сегменты")
            return []

        print(f"\n  ✓ Выделено сегментов: {len(segments)}")

        print("\n  🔍 Этап 2: Генерация саммари")
        segments = self.generate_summaries(segments)

        topics = []
        for i, seg in enumerate(segments, 1):
            topics.append(Topic(
                number=i,
                start_time=self._format_time(seg['start']),
                end_time=self._format_time(seg['end']),
                summary=seg.get('summary', 'Тема не определена'),
                full_text=seg['text']
            ))

        return topics

    def _parse_json_response(self, response: str) -> Optional[List[Dict]]:
        """Парсинг JSON из ответа LLM"""
        try:
            response = response.strip()

            start = response.find('[')
            end = response.rfind(']') + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
                if self.debug:
                    print(f"\n    ✓ JSON распарсен: {len(parsed)} элементов")
                return parsed

            return json.loads(response)
        except Exception as e:
            if self.debug:
                print(f"\n    ✗ Ошибка парсинга JSON: {e}")
            return None

    def _format_time(self, seconds: float) -> str:
        """Форматирование времени MM:SS"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def save_csv(self, topics: List[Topic], output_path: Path):
        """Сохранение в CSV"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Номер', 'Начало', 'Конец', 'Саммари', 'Полный текст'
            ], delimiter=';')

            writer.writeheader()

            for topic in topics:
                writer.writerow({
                    'Номер': topic.number,
                    'Начало': topic.start_time,
                    'Конец': topic.end_time,
                    'Саммари': topic.summary.replace('\n', ' '),
                    'Полный текст': topic.full_text.replace('\n', ' ')
                })

    def process_file(self, input_file: Path) -> Optional[Path]:
        """Обработка файла"""
        print(f"\n{'='*60}")
        print(f"📂 Обработка: {input_file.name}")
        print(f"{'='*60}")

        lines = self.parse_transcript(input_file)

        if not lines:
            print("  ✗ Не удалось распарсить файл")
            return None

        topics = self.analyze(lines)

        if not topics:
            print("  ✗ Темы не выделены")
            return None

        print(f"\n  ✅ Итого тем: {len(topics)}")

        output_dir = Path(self.config.get('paths', 'output_dir', default='results'))
        output_path = output_dir / f"{input_file.stem}_analysis.csv"

        self.save_csv(topics, output_path)
        print(f"  💾 Сохранено: {output_path}\n")

        return output_path


def main():
    import sys

    if len(sys.argv) < 2:
        print("Использование:")
        print("  python analyzer.py <файл.txt>")
        print("  python analyzer.py --batch <папка>")
        return

    analyzer = TranscriptAnalyzer("config.yaml")

    if sys.argv[1] == "--batch":
        input_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('transcripts')

        if not input_dir.is_dir():
            print(f"Ошибка: {input_dir} не является папкой")
            return

        files = list(input_dir.glob("*.txt"))
        print(f"Найдено файлов: {len(files)}")

        for file in files:
            try:
                analyzer.process_file(file)
            except Exception as e:
                print(f"Ошибка при обработке {file.name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        input_file = Path(sys.argv[1])

        if not input_file.exists():
            print(f"Ошибка: файл {input_file} не найден")
            return

        analyzer.process_file(input_file)


if __name__ == "__main__":
    main()