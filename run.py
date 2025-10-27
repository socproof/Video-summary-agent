#!/usr/bin/env python
"""
Скрипт для запуска анализатора транскрипций
Поддерживает обработку одного файла или всей папки
"""

import sys
from pathlib import Path
from transcript_analyzer import TranscriptAnalyzer


def process_single_file(file_path: str, config_path: str = "config.yaml"):
    """Обработка одного файла"""
    analyzer = TranscriptAnalyzer(config_path)
    input_file = Path(file_path)

    if not input_file.exists():
        print(f"❌ Ошибка: файл не найден - {file_path}")
        return False

    if input_file.suffix != '.txt':
        print(f"❌ Ошибка: ожидается .txt файл, получен {input_file.suffix}")
        return False

    try:
        output_file = analyzer.process_file(input_file)
        if output_file:
            print(f"✅ Успешно! Результат: {output_file}")
            return True
        else:
            print(f"⚠️  Предупреждение: не удалось обработать файл")
            return False
    except Exception as e:
        print(f"❌ Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_batch(directory: str, config_path: str = "config.yaml"):
    """Пакетная обработка всех файлов в папке"""
    analyzer = TranscriptAnalyzer(config_path)
    input_dir = Path(directory)

    if not input_dir.exists():
        print(f"❌ Ошибка: папка не найдена - {directory}")
        return

    if not input_dir.is_dir():
        print(f"❌ Ошибка: {directory} не является папкой")
        return

    # Находим все .txt файлы
    txt_files = list(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"⚠️  Не найдено .txt файлов в {directory}")
        return

    print(f"📁 Найдено файлов для обработки: {len(txt_files)}\n")

    success_count = 0
    failed_count = 0

    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}] " + "=" * 50)
        try:
            output_file = analyzer.process_file(txt_file)
            if output_file:
                success_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            failed_count += 1

    # Итоговая статистика
    print("\n" + "=" * 50)
    print(f"📊 Итого:")
    print(f"   ✅ Успешно обработано: {success_count}")
    print(f"   ❌ Ошибок: {failed_count}")
    print(f"   📁 Всего файлов: {len(txt_files)}")


def print_usage():
    """Вывод справки по использованию"""
    print("""
Анализатор транскрипций видео
=============================

Использование:
  python run.py <файл.txt>              - Обработать один файл
  python run.py --batch <папка>         - Обработать все .txt файлы в папке
  python run.py --config <config.yaml>  - Указать путь к конфигу

Примеры:
  python run.py transcript.txt
  python run.py --batch ./transcripts
  python run.py transcript.txt --config my_config.yaml
  python run.py --batch ./data --config custom.yaml

Формат входного файла:
  [SPEAKER_00] 0.0-10.5: Текст реплики
  или
  0.0-10.5: Текст реплики
  или
  [0.0 - 10.5] Текст реплики

Результат:
  CSV файл с колонками: Номер, Начало, Конец, Саммари, Полный текст
""")


def main():
    """Главная функция"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Парсинг аргументов
    args = sys.argv[1:]
    config_path = "config.yaml"
    mode = None
    target = None

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ['-h', '--help', 'help']:
            print_usage()
            sys.exit(0)

        elif arg == '--config':
            if i + 1 < len(args):
                config_path = args[i + 1]
                i += 2
            else:
                print("❌ Ошибка: не указан путь к конфигу")
                sys.exit(1)

        elif arg == '--batch':
            mode = 'batch'
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                target = args[i + 1]
                i += 2
            else:
                target = './transcripts'  # По умолчанию
                i += 1

        elif not arg.startswith('--'):
            if mode is None:
                mode = 'single'
                target = arg
            i += 1

        else:
            print(f"❌ Неизвестный параметр: {arg}")
            print_usage()
            sys.exit(1)

    # Проверка конфига
    if not Path(config_path).exists():
        print(f"⚠️  Внимание: конфиг {config_path} не найден, используются настройки по умолчанию")

    # Выполнение
    if mode == 'batch':
        process_batch(target, config_path)
    elif mode == 'single':
        success = process_single_file(target, config_path)
        sys.exit(0 if success else 1)
    else:
        print("❌ Ошибка: не указан файл или папка для обработки")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()