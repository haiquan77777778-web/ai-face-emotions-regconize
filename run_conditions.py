"""

Nhóm:
    A. Cơ sở: 7 biểu cảm (happy, sad, angry, surprise, fear, disgust, neutral) với MediaPipe + EfficientNet
    B. Ánh sáng: thiếu sáng + ngược sáng, mỗi loại 3 biểu cảm (happy, surprise, neutral)
    E. Khoảng cách: 2 m và 3 m, mỗi mức 3 biểu cảm (happy, surprise, neutral)

Enter = quay, s = bỏ qua lượt này, q = dừng (chạy lại sau sẽ tiếp tục, lượt đã xong được bỏ qua).
Mỗi lượt gọi condition_test.py, kết quả nối thêm vào conditions_log.csv. Chỉ lưu số liệu, không lưu hình.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS = os.path.join(HERE, "results", "conditions_progress.txt")
ALL7 = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
KEY3 = ["happy", "surprise", "neutral"]


def build_plan(group, kinh=False):
    """Trả về danh sách (khóa, mô tả, [tham số cho condition_test.py])."""
    plan = []
    if group == "A":
        occl = "kinh" if kinh else "none"
        tag = "deo kinh" if kinh else "binh thuong"
        for e in ALL7:
            plan.append((f"A|{occl}|mediapipe|{e}", f"{e}, {tag}, mediapipe",
                         ["--expr", e, "--occl", occl]))
    elif group == "B":
        for light in ("thieu_sang", "nguoc_sang"):
            for e in KEY3:
                plan.append((f"B|{light}|{e}", f"{e}, {light}",
                             ["--expr", e, "--light", light]))
    elif group == "E":
        for d in ("2", "3"):
            for e in KEY3:
                plan.append((f"E|{d}m|{e}", f"{e}, cach {d} m",
                             ["--expr", e, "--dist", d]))
    return plan


def load_done():
    if not os.path.exists(PROGRESS):
        return set()
    with open(PROGRESS, encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", default=None, help="tên người thử")
    ap.add_argument("--groups", required=True, choices=["A", "B", "E"])
    ap.add_argument("--kinh", action="store_true", help="nhóm A khi đeo kính")
    ap.add_argument("--countdown", type=float, default=3.0, help="số giây đếm ngược trước khi quay")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    a.model = "efficientnet"
    if a.kinh and a.groups != "A":
        sys.exit("--kinh chỉ dùng với --groups A")

    plan = build_plan(a.groups, a.kinh)
    if a.list:
        for i, (_, d, _a) in enumerate(plan, 1):
            print(f"{i:2d}. {d}")
        print(f"Tong {len(plan)} luot")
        return
    if not a.who:
        sys.exit("Can --who TEN. Vi du: python run_conditions.py --who An --groups B")

    done = load_done()
    cond = os.path.join(HERE, "scripts", "condition_test.py")
    todo = [i for i, p in enumerate(plan) if f"{a.who}|{a.model}|{p[0]}" not in done]
    print(f"Nguoi thu: {a.who} | model: {a.model} | nhom {a.groups}{' (kinh)' if a.kinh else ''} | con {len(todo)}/{len(plan)} luot")
    print("Enter = quay, s = bo qua luot nay, q = dung (chay lai sau se tiep tuc)\n")
    for i in todo:
        key, d, args = plan[i]
        print(f"--- Luot {i + 1}/{len(plan)}: {d}")
        cmd = input("   Chuan bi xong chua? (Enter/s/q) ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        rc = subprocess.call([sys.executable, cond, "--who", a.who,
                              "--countdown", str(a.countdown)] + args)
        if rc == 0:
            with open(PROGRESS, "a", encoding="utf-8") as f:
                f.write(f"{a.who}|{a.model}|{key}\n")
        else:
            print("   (loi, luot nay chua duoc danh dau xong)")
    print("\nXong. Tong hop ket qua:  python summarize_conditions.py")


if __name__ == "__main__":
    main()
