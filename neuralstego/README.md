# neuralstego

`neuralstego` یک ابزار خط فرمان (CLI) پایتونی برای کار با پروژه‌های نهان‌نگاری عصبی است. در این مرحله ساختار اولیه پروژه آماده شده است.

## پیش‌نیازها

- Python 3.10 یا بالاتر
- pip
- (اختیاری) virtualenv یا ابزار مشابه برای ساخت محیط مجزا

## نصب محلی

```bash
python -m venv .venv
source .venv/bin/activate  # در ویندوز: .venv\\Scripts\\activate
python -m pip install --upgrade pip

# از ریشه مخزن (جایی که فایل pyproject.toml قرار دارد)
python -m pip install -e .
```

> **نکته:** اگر از shellهایی مثل `zsh` استفاده می‌کنید، برای نصب extras باید عبارت را کوتیشن‌گذاری کنید تا glob نشود؛ نمونه:
> 
> ```bash
> python -m pip install -e '.[dev]'
> ```

## اجرای ابزار CLI

پس از نصب محلی، می‌توانید فرمان کمک را اجرا کنید:

```bash
python -m neuralstego --help
```

یا با اسکریپت نصب‌شده:

```bash
neuralstego --help
```

## تست دود (Smoke Test)

یک تست ساده برای اطمینان از آماده بودن محیط وجود دارد:

```bash
pytest -q
```

برای اجرای اسکریپت دود CLI (پس از دادن مجوز اجرا):

```bash
chmod +x scripts/smoke_test_cli.sh
./scripts/smoke_test_cli.sh
```

## اسکریپت‌ها

پوشه `scripts/` شامل اسکریپت‌های کمکی است. برای اجرای آن‌ها ابتدا مجوز اجرا بدهید:

```bash
chmod +x scripts/*.sh
```

سپس مثلاً:

```bash
./scripts/doctor.sh
```

## توسعه

- برای نصب وابستگی‌های توسعه:

  ```bash
  python -m pip install -e '.[dev]'
  ```

- اجرای بررسی‌های CI محلی:

  ```bash
  pytest -q
  ```

## مجوز

این پروژه تحت مجوز MIT منتشر شده است. متن کامل مجوز در فایل [LICENSE](LICENSE) موجود است.
