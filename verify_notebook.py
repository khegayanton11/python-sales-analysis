"""Прогон всех кодовых ячеек ноутбука в одном пространстве имён.

Исполняет весь код ноутбука и сверяет ключевые значения с независимым
расчётом на исходных данных. Расхождение любого числа роняет проверку.
"""
import json, sys, io, traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nb = json.load(open("sales_analysis.ipynb", encoding="utf-8"))
code_cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
              if c["cell_type"] == "code"]

ns, failed = {}, []
for i, src in code_cells:
    src = src.replace('pd.read_csv("data/data.csv")', 'pd.read_csv("data/data.csv")')
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        exec(compile(src, f"<ячейка {i}>", "exec"), ns); ok = True
    except Exception:
        ok = False; tb = traceback.format_exc()
    finally:
        sys.stdout = old; plt.close("all")
    out = buf.getvalue().strip()
    if ok:
        print(f"[OK ] ячейка {i:2d}: {out.split(chr(10))[0][:66] if out else '(без текстового вывода)'}")
    else:
        failed.append(i); print(f"[FAIL] ячейка {i:2d}\n{tb}")

print("=" * 62)
print(f"выполнено ячеек: {len(code_cells)}, ошибок: {len(failed)}")
assert not failed, failed

print("\n=== СВЕРКА ЧИСЕЛ ===")
df = pd.read_csv("data/data.csv"); df["Дата"] = pd.to_datetime(df["Дата"])
g = df.groupby("Дата")["Количество"].sum()
errs = []
def chk(name, got, exp, tol=0.01):
    ok = abs(float(got) - float(exp)) <= tol
    print(f"  [{'OK ' if ok else 'ОШИБКА'}] {name}: {got} / {exp}")
    if not ok: errs.append(name)

chk("строк", len(ns["df"]), 301355, 0)
chk("уникальных дат", ns["df"]["Дата"].nunique(), 205, 0)
chk("строк в grouped_df", len(ns["grouped_df"]), 205, 0)
chk("сумма продаж", ns["grouped_df"]["Количество продаж"].sum(), df["Количество"].sum(), 0)
chk("максимум в строке", ns["outlier_row"]["Количество"], 200, 0)
chk("z-оценка дня 28.06", ns["z"], (g.loc["2018-06-28"]-g.mean())/g.std(), 0.01)
chk("точка перелома = 24.04", ns["best_date"].dayofyear, pd.Timestamp("2018-04-24").dayofyear, 0)
chk("t-статистика май vs янв-апр", ns["t_stat"], -13.64, 0.05)
chk("топ-товар среза = product_1", 1 if ns["answer"] == "product_1" else 0, 1, 0)
chk("продано в срезе", ns["subset"]["Количество"].sum(), 10517, 0)
chk("товаров в группе A", (ns["abc"]["группа"] == "A").sum(), 7, 0)
chk("доля топ-10 клиентов", ns["clients"].head(10).sum()/ns["clients"].sum(), 0.1352, 0.001)
chk("пропущено дней", len(ns["missing"]), 35, 0)
chk("строк-выбросов по IQR", ns["n_out"], 13140, 0)

w = pd.read_csv("data/weather_astana_2018.csv", parse_dates=["Дата"])
mm = df.groupby("Дата")["Количество"].sum().reset_index()
mm.columns = ["Дата", "s"]
mm = mm.merge(w, on="Дата").dropna()
mm["мес"] = mm["Дата"].dt.to_period("M")
mm["sd"] = mm["s"] - mm.groupby("мес")["s"].transform("mean")
mm["td"] = mm["T"] - mm.groupby("мес")["T"].transform("mean")
from scipy import stats as _s
chk("дней с температурой", len(mm), 205, 0)
chk("r сырых рядов", ns["r_raw"], _s.pearsonr(mm["T"], mm["s"])[0], 0.0001)
chk("r отклонений", ns["r_adj"], _s.pearsonr(mm["td"], mm["sd"])[0], 0.0001)
chk("r плацебо", ns["r_placebo"], _s.pearsonr(np.arange(len(mm)), mm["s"])[0], 0.0001)
chk("r сырых рядов = 0.60", ns["r_raw"], 0.6038, 0.001)
chk("r отклонений = -0.06", ns["r_adj"], -0.0611, 0.001)
chk("r плацебо = 0.69", ns["r_placebo"], 0.6903, 0.001)
chk("частная корреляция = -0.08", ns["pr"], -0.0844, 0.001)

print(f"\nрасхождений: {len(errs)}")
assert not errs, errs
print("ВСЕ ЯЧЕЙКИ ИСПОЛНЯЮТСЯ, ЧИСЛА СОВПАДАЮТ")

