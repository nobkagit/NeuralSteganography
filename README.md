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
۱. دریافت مخزن و ورود به آن:

```bash
git clone https://github.com/example/neuralsteganography.git
cd NeuralSteganography
```

۲. ساخت محیط مجازی و نصب وابستگی‌ها (به‌همراه requirements.txt در صورت وجود):

```bash
make init
source .venv/bin/activate
```

> در صورت تمایل می‌توانید مستقیماً اسکریپت را اجرا کنید: `bash scripts/setup_env.sh`

۳. دریافت تنبل مدل فارسی پیش‌فرض و مشاهدهٔ مسیر کش:

```bash
python scripts/download_models.py --model HooshvareLab/gpt2-fa
```

۴. بررسی سلامت ابزارهای سیستم و وابستگی‌های Python:

```bash
bash scripts/doctor.sh
neuralstego doctor
```

۵. اجرای تست دود CLI و pytest:

```bash
make smoke
```

> اسکریپت smoke از داخل `.venv` اجرا می‌شود و دستورات `neuralstego --version`، `neuralstego doctor` و `pytest -q` را بررسی می‌کند.

## استفاده از دستورات Make
دستورات پرکاربرد برای توسعه‌دهندگان:

```bash
make init   # ساخت/به‌روزرسانی محیط مجازی و نصب وابستگی‌ها
make doctor # اجرای scripts/doctor.sh
make test   # اجرای pytest -q در داخل venv
make lint   # اجرای ruff check .
make type   # اجرای mypy روی src
make smoke  # اجرای اسکریپت smoke (نسخه، doctor، pytest)
```

## آزمون دود پوششی (Cover Smoke)
برای آزمایش سریع جریان cover می‌توانید اسکریپت‌های دود زیر را اجرا کنید:

```bash
bash scripts/cover_smoke.sh          # دود کلاسیک بدون نگهبان کیفیت
bash scripts/quality_smoke.sh        # دود همراه با gate و audit
bash scripts/audit_many.sh           # تولید مجموعه‌ای از کاورها و ممیزی گروهی
bash scripts/ui_smoke.sh             # اطمینان از بالا آمدن رابط Gradio
```

این اسکریپت‌ها پیام‌های محرمانهٔ نمونه می‌سازند، آن‌ها را در متون پوششی جاسازی می‌کنند و نتیجهٔ نگهبان کیفیت و ممیزی را گزارش می‌دهند. در صورت نیاز به اجرای دستی، می‌توانید از دستورات زیر (با به‌روزرسانی مسیرها و seed) استفاده کنید:

```bash
neuralstego cover-generate -p "Pa$$w0rd" -i secret.txt -o cover.txt \
  --seed "در یک گفت‌وگوی کوتاه درباره‌ی فناوری و اخبار روز صحبت می‌کنیم." \
  --quality top-k 60 --quality temperature 0.8 \
  --quality-gate on --max-ppl 100 --max-ngram-repeat 0.25 --min-ttr 0.30 \
  --regen-attempts 2 --chunk-bytes 256 --crc on --ecc rs --nsym 10

neuralstego cover-reveal -p "Pa$$w0rd" -i cover.txt -o recovered.txt \
  --seed "در یک گفت‌وگوی کوتاه درباره‌ی فناوری و اخبار روز صحبت می‌کنیم."

neuralstego quality-audit -i cover.txt --max-ppl 100 --max-ngram-repeat 0.25 --min-ttr 0.30
```

## رابط کاربری وب Gradio
برای کاربرانی که رابط تعاملی ترجیح می‌دهند، ماژول Gradio در دسترس است. پیش از اجرا اطمینان حاصل کنید که وابستگی‌ها (به‌ویژه `gradio>=4.0.0`) نصب شده‌اند.

```bash
neuralstego ui --port 7860 --share false
# یا
bash scripts/run_gradio.sh
```

گزینهٔ `--device` و متغیر محیطی `NEURALSTEGO_DEVICE` اجازه می‌دهند بین `auto`، `cpu` یا `cuda` انتخاب کنید. اگر نیاز به بررسی سریع بالا آمدن سرویس دارید، اسکریپت `bash scripts/ui_smoke.sh` سرور را برای چند ثانیه اجرا کرده و پیام موفقیت چاپ می‌کند.

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
- تکمیل `scripts/download_models.py` با کنترل نسخه و گزارش دقیق ظرفیت کش.
- پیاده‌سازی کامل لایهٔ encode/decode با رمزگذار حسابی و الگوهای تولید متن.

با دنبال‌کردن مراحل فوق باید بتوانید محیط توسعه را در چند دقیقه آماده کنید.
