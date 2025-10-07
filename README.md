# NeuralSteganography

[![CI](https://img.shields.io/badge/CI-pending-lightgrey)](https://github.com/example/neuralsteganography/actions)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%26%20mypy-lightgrey)](https://github.com/example/neuralsteganography/actions)

## معرفی کوتاه
NeuralSteganography نسخهٔ مدرن و توسعه‌یافته‌ای از الگوریتم‌های استگانوگرافی متنی است که روی مدل‌های زبانی نظیر GPT2-fa و رمزگذارهای حسابی تکیه دارد. این مخزن پایهٔ موردنیاز برای فازهای بعدی پروژهٔ «NeuralStego» را آماده می‌کند؛ شامل CLI، اسکریپت‌های راه‌اندازی و راهنمای عامل‌ها.

## پیش‌نیازها
- Python ≥ 3.10
- ابزارهای خط فرمان استاندارد (bash، make، git)
- دسترسی به اینترنت برای دریافت مدل‌ها از HuggingFace

## نصب و راه‌اندازی سریع
۱. ساخت مخزن و محیط مجازی:

```bash
git clone https://github.com/example/neuralsteganography.git
cd NeuralSteganography
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

۲. آماده‌سازی فایل‌های محیطی و پوشه‌ها:

```bash
bash scripts/setup_env.sh
```

۳. دریافت مدل‌ها (پیش‌فرض: `HooshvareLab/gpt2-fa`):

```bash
python scripts/download_models.py --target models
```

۴. بررسی سلامت محیط:

```bash
neuralstego doctor
```

۵. اجرای تست دود CLI:

```bash
bash scripts/smoke_test_cli.sh
```

> **نکته:** اگر از ویندوز استفاده می‌کنید، معادل‌های PowerShell (`.\venv\Scripts\Activate.ps1`) و اجرای اسکریپت‌ها از طریق `bash` در WSL پیشنهاد می‌شود.

## استفاده از دستورات Make
برای خودکارسازی مراحل بالا می‌توانید از دستورات زیر استفاده کنید:

```bash
make install   # ایجاد venv و نصب editable
make setup     # اجرای scripts/setup_env.sh
make doctor    # اجرای neuralstego doctor داخل venv
make smoke     # اجرای smoke_test_cli.sh پس از doctor
```

## ساختار معماری و فازها
- **فاز ۰: اسکلت پروژه** — تنظیم ساختار پوشه‌ها، ابزارهای توسعه، راهنمای عامل‌ها.
- **فاز ۱: CLI و زیرسیستم‌ها** — پیاده‌سازی دستورات encode/decode و زیرساخت‌های دکتورینگ.
- **فاز ۲: ادغام مدل‌های NLP** — اتصال GPT2-fa، مدیریت Tokenizer و بهینه‌سازی استنتاج.
- **فاز ۳: رمزنگاری و امنیت** — افزودن لایه‌های AES/AEAD و کنترل کلیدها.
- **فاز ۴: کیفیت و استقرار** — تست‌های پیشرفته، CI/CD، بسته‌بندی و انتشار.

در هر فاز، تعریف Done شامل گذر از تست‌ها (unit/cli)، lint (`ruff`) و بررسی type (`mypy`) به همراه به‌روزرسانی مستندات است.

## رمزنگاری مبتنی بر رمز عبور
CLI اکنون شامل زیر‌فرمان‌های `encrypt` و `decrypt` است که پیام‌ها را با AES-GCM و KDF پیکربندی‌پذیر در قالب JSON ذخیره می‌کنند.

```bash
# رمزگذاری فایل متنی با رمز عبور و نوشتن خروجی باینری
neuralstego encrypt -p "Pa$$w0rd" -i message.txt -o message.enc

# رمزگشایی envelope و بازگرداندن متن اصلی
neuralstego decrypt -p "Pa$$w0rd" -i message.enc -o recovered.txt
```

هر دو فرمان از `-` برای stdin/stdout پشتیبانی می‌کنند و در صورت نوشتن باینری روی ترمینال اخطار می‌دهند. در صورت عدم تعیین گزینهٔ `--password`، رمز عبور به‌صورت امن از کاربر پرسیده می‌شود.

## مسیرهای بعدی
- مطالعهٔ فایل [AGENT.md](AGENT.md) برای راهنمای پرامت‌دهی عامل‌ها.
- تکمیل `scripts/download_models.py` با پشتیبانی از کش و کنترل نسخهٔ مدل.
- پیاده‌سازی کامل لایهٔ encode/decode با رمزگذار حسابی و الگوهای تولید متن.

با دنبال‌کردن مراحل فوق باید بتوانید محیط توسعه را در چند دقیقه آماده کنید.
