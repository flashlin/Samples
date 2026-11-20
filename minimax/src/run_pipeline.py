#!/usr/bin/env python3
"""
MiniMax H3 Ref2VA HyperFlow Flash (6步參考圖影音生成) + SeedVR2 放大 主執行程式
功能：
1. 讀取專案根目錄下的 prompt.txt 提示詞檔案
2. 自動讀取人物定裝參考圖：
   - 固定檢查專案目錄下的 person1.jpg、person2.jpg（若存在則自動載入，不存在則忽略跳過）
   - 若提示詞中包含 <person1.jpg> 或其他 <檔名.jpg/png/webp> 標籤，自動載入對應圖檔並轉換為 <Picture N>；找不到檔案則忽略跳過
3. 詢問使用者：
   - 影片解析度 (預設 1024x768；若大於 1024x768 則先以 1024x768 生成後自動用 SeedVR2 3B 放大)
   - 影片秒數 (預設 15 秒；超過單段 5 秒時，自動擷取上一段最後一格影格作為開頭錨點，並持續套用人物參考圖接續生成)
4. 透過 comfy-cli (或本地 ComfyUI Server) 啟動並執行 HyperFlow Ref2VA 6步工作流
5. 產生期間即時顯示 tqdm 進度條 (節點進度與步數進度)
6. 將所有分段影音無縫合併並固定輸出至 outputs/ 目錄
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v3 as iio
import imageio_ffmpeg
from PIL import Image
from tqdm import tqdm
import websocket

from apply_mac_optimizations import patch_hyperflow_gguf
from download_models import ensure_models
from workflow_builder import (
    CANVAS_MULTIPLE,
    DEFAULT_REF_IMAGE_FILES,
    build_hyperflow_workflow,
    parse_and_resolve_references,
    seconds_to_h3_frames,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
COMFY_DIR = ROOT_DIR / "ComfyUI"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PROMPT_FILE = ROOT_DIR / "prompt.txt"
WORKFLOWS_TMP_DIR = ROOT_DIR / "outputs" / ".workflows"

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_DURATION = 15.0
SEGMENT_MAX_SECONDS = 5.0  # 單段生成單位 (5秒 = 120/124 frames)，超過自動接續


def get_ffmpeg_exe() -> str:
    local_bin = ROOT_DIR / "bin" / "ffmpeg"
    if local_bin.exists():
        return str(local_bin)
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_prompt_file(prompt_path: Path) -> str:
    if not prompt_path.exists():
        default_text = (
            "A cinematic close-up of a futuristic neon-lit street in Tokyo during a gentle rain at night, "
            "reflections glistening on wet pavement, ambient synthwave music and soft rain sounds in the background, "
            "slow camera pan."
        )
        prompt_path.write_text(default_text + "\n", encoding="utf-8")
        print(f"📝 已自動建立預設提示詞檔案: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"提示詞檔案 {prompt_path} 內容為空，請填入提示詞後再試！")
    return text


def parse_resolution(res_str: str) -> Tuple[int, int]:
    res_str = res_str.strip().lower().replace("*", "x").replace("×", "x")
    if not res_str:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    if "x" not in res_str:
        raise ValueError(f"解析度格式錯誤 '{res_str}'，請輸入如 1024x768 或 1920x1080")
    parts = res_str.split("x")
    w, h = int(parts[0].strip()), int(parts[1].strip())
    if w < 256 or h < 256:
        raise ValueError("解析度長寬至少需 256 以上")
    return w, h


def ask_user_parameters(
    cli_resolution: Optional[str] = None,
    cli_duration: Optional[float] = None,
    cli_steps: int = 6,
    non_interactive: bool = False,
) -> Tuple[int, int, float, int]:
    print("\n========================================================")
    print("🎬 MiniMax H3 Ref2VA HyperFlow Flash (6步參考圖影音生成) + SeedVR2 放大")
    print("========================================================")

    if cli_resolution:
        width, height = parse_resolution(cli_resolution)
    elif non_interactive:
        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    else:
        try:
            res_input = input(
                f"👉 請輸入要產生的畫面大小 [預設 {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}，若更大將自動啟用 SeedVR2 放大]: "
            ).strip()
        except EOFError:
            res_input = ""
        width, height = parse_resolution(res_input) if res_input else (DEFAULT_WIDTH, DEFAULT_HEIGHT)

    if cli_duration is not None:
        duration = float(cli_duration)
    elif non_interactive:
        duration = DEFAULT_DURATION
    else:
        try:
            dur_input = input(
                f"👉 請輸入要跑多久秒數 [預設 {int(DEFAULT_DURATION)} 秒，超過 {int(SEGMENT_MAX_SECONDS)} 秒自動接續最後一格]: "
            ).strip()
        except EOFError:
            dur_input = ""
        duration = float(dur_input) if dur_input else DEFAULT_DURATION

    if duration <= 0:
        raise ValueError("影片秒數必須大於 0 秒")

    return width, height, duration, cli_steps


def is_server_running(host: str = "127.0.0.1", port: int = 8188) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/system_stats", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_comfy_server_via_cli(host: str = "127.0.0.1", port: int = 8188) -> Optional[subprocess.Popen]:
    if is_server_running(host, port):
        print(f"✅ 偵測到 ComfyUI Server 已在 http://{host}:{port} 運行中")
        return None

    comfy_bin = ROOT_DIR / ".venv" / "bin" / "comfy"
    python_bin = ROOT_DIR / ".venv" / "bin" / "python"
    log_file = OUTPUTS_DIR / "comfy_server.log"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PATH"] = f"{ROOT_DIR / 'bin'}:{ROOT_DIR / '.venv' / 'bin'}:{env.get('PATH', '')}"

    print(f"🚀 正在透過 comfy-cli 啟動本地 Mac ComfyUI Server (Port {port})...")
    log_fp = open(log_file, "a", encoding="utf-8")

    # 優先使用 comfy-cli launch，並開啟 Apple Silicon (MPS) 原生 PyTorch SDPA Attention 與 HighVRAM 模式
    extra_comfy_args = [
        "--listen",
        host,
        "--port",
        str(port),
        "--use-pytorch-cross-attention",
        "--highvram",
    ]
    cmd = [
        str(comfy_bin),
        f"--workspace={COMFY_DIR}",
        "launch",
        "--",
        *extra_comfy_args,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(COMFY_DIR),
        env=env,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
    )

    # 等待伺服器就緒 (若 comfy-cli 因 sandbox 無法寫入 ~/Library，自動切換為直接啟動 main.py)
    start_time = time.time()
    with tqdm(total=60, desc="⏳ 等待 ComfyUI 引擎啟動", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s") as pbar:
        while time.time() - start_time < 60:
            if is_server_running(host, port):
                pbar.n = pbar.total
                pbar.refresh()
                print(f"✅ ComfyUI Server 啟動成功 (已啟用 Metal SDPA + GGUF 權重快取加速)！")
                return proc

            if proc.poll() is not None:
                # comfy-cli 異常退出，改用 python main.py 直接啟動
                cmd_fallback = [
                    str(python_bin),
                    str(COMFY_DIR / "main.py"),
                    *extra_comfy_args,
                ]
                proc = subprocess.Popen(
                    cmd_fallback,
                    cwd=str(COMFY_DIR),
                    env=env,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                )

            time.sleep(1)
            pbar.update(1)

    raise RuntimeError(f"ComfyUI 伺服器啟動超時，請檢查日誌: {log_file}")


def execute_workflow_with_progress(
    workflow: dict,
    segment_idx: int,
    total_segments: int,
    host: str = "127.0.0.1",
    port: int = 8188,
) -> Path:
    """
    提交工作流至 ComfyUI API，並透過 WebSocket 即時顯示節點與采样進度條，最後回傳產生的 MP4 路徑。
    """
    client_id = str(uuid.uuid4())
    ws_url = f"ws://{host}:{port}/ws?clientId={client_id}"
    ws = websocket.WebSocket()
    ws.connect(ws_url, timeout=30)
    # 推論期間 (如 Qwen3-VL 32B 編碼、14B DiT 第一步 MPS 編譯與採樣、VAE 解碼) 單步可能超過 30 秒，
    # 將 recv timeout 設為 600 秒並在捕獲 WebSocketTimeoutException 時自動輪詢伺服器狀態繼續等待
    ws.settimeout(600)

    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            prompt_id = resp_data["prompt_id"]
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        ws.close()
        raise RuntimeError(f"ComfyUI 工作流提交失敗 ({e.code}): {err_body}")

    total_nodes = len(workflow)
    completed_nodes = 0
    current_node_title = "初始化"

    node_pbar = tqdm(
        total=total_nodes,
        desc=f"🎬 段落 [{segment_idx}/{total_segments}] 節點進度",
        position=0,
        leave=True,
    )
    step_pbar = tqdm(
        total=6,
        desc=f"⚡ 段落 [{segment_idx}/{total_segments}] 採樣步數 ({current_node_title})",
        position=1,
        leave=False,
    )

    try:
        while True:
            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                # 若單步推論時間較長，檢查是否已完成或伺服器仍在執行中
                try:
                    with urllib.request.urlopen(f"http://{host}:{port}/history/{prompt_id}", timeout=5) as resp:
                        hist_check = json.loads(resp.read().decode("utf-8"))
                    if prompt_id in hist_check:
                        node_pbar.n = node_pbar.total
                        node_pbar.refresh()
                        break
                except Exception:
                    pass
                continue
            if isinstance(out, bytes):
                continue
            msg = json.loads(out)
            msg_type = msg.get("type")
            data = msg.get("data", {})

            if data.get("prompt_id") and data.get("prompt_id") != prompt_id:
                continue

            if msg_type == "executing":
                node_id = data.get("node")
                if node_id is None:
                    # 執行完成
                    node_pbar.n = node_pbar.total
                    node_pbar.refresh()
                    break
                node_info = workflow.get(str(node_id), {})
                current_node_title = node_info.get("_meta", {}).get("title") or node_info.get("class_type", f"Node {node_id}")
                node_pbar.set_postfix_str(f"正在執行: {current_node_title}")
                step_pbar.set_description(f"⚡ 段落 [{segment_idx}/{total_segments}] {current_node_title}")
                step_pbar.reset(total=100)

            elif msg_type == "progress":
                val = data.get("value", 0)
                max_val = data.get("max", 100)
                if step_pbar.total != max_val:
                    step_pbar.total = max_val
                step_pbar.n = val
                step_pbar.refresh()

            elif msg_type == "executed":
                completed_nodes += 1
                if completed_nodes <= total_nodes:
                    node_pbar.n = completed_nodes
                    node_pbar.refresh()

            elif msg_type == "execution_error":
                exc_msg = data.get("exception_message", "Unknown execution error")
                node_type = data.get("node_type", "UnknownNode")
                raise RuntimeError(f"ComfyUI 節點 [{node_type}] 執行錯誤: {exc_msg}")

    finally:
        step_pbar.close()
        node_pbar.close()
        ws.close()

    # 從 /history/{prompt_id} 取得產生的影片檔案
    with urllib.request.urlopen(f"http://{host}:{port}/history/{prompt_id}") as resp:
        history = json.loads(resp.read().decode("utf-8"))

    outputs = history.get(prompt_id, {}).get("outputs", {})
    for node_id, node_out in outputs.items():
        for key in ("videos", "gifs", "images"):
            if key in node_out:
                for item in node_out[key]:
                    filename = item.get("filename")
                    subfolder = item.get("subfolder", "")
                    if filename and filename.lower().endswith((".mp4", ".webm", ".mov")):
                        video_path = COMFY_DIR / "output" / subfolder / filename
                        if video_path.exists():
                            return video_path

    # 若 history 格式不同，搜尋 ComfyUI/output/ 下最新產生的 mp4
    output_root = COMFY_DIR / "output"
    mp4_files = sorted(output_root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if mp4_files:
        return mp4_files[0]

    raise FileNotFoundError("工作流執行完畢，但在 ComfyUI/output 中找不到輸出的 MP4 影片檔！")


def extract_last_frame(video_path: Path, dest_image_path: Path) -> Path:
    """
    從上一段生成的影片中擷取最後一格影格，並存入 ComfyUI/input/ 供下一段 I2V 接續使用。
    """
    dest_image_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = get_ffmpeg_exe()

    # 使用 ffmpeg 擷取倒數第一格影格
    cmd = [
        ffmpeg_exe,
        "-y",
        "-sseof",
        "-0.1",
        "-i",
        str(video_path),
        "-update",
        "1",
        "-q:v",
        "1",
        str(dest_image_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    if not dest_image_path.exists() or dest_image_path.stat().st_size == 0:
        # Fallback: 使用 imageio 讀取最後一格
        frames = iio.imread(str(video_path), plugin="pyav" if False else None)
        last_frame = frames[-1]
        Image.fromarray(last_frame).save(str(dest_image_path))

    return dest_image_path


def concatenate_segments(segment_paths: List[Path], final_output_path: Path) -> Path:
    """
    將多段生成的有聲影片無縫接續合併為最終長影片，輸出至 outputs/。
    """
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(segment_paths) == 1:
        shutil.copy2(str(segment_paths[0]), str(final_output_path))
        return final_output_path

    ffmpeg_exe = get_ffmpeg_exe()
    concat_list_file = final_output_path.parent / f".concat_{final_output_path.stem}.txt"
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for seg in segment_paths:
            f.write(f"file '{seg.resolve()}'\n")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(final_output_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if concat_list_file.exists():
        concat_list_file.unlink()

    if res.returncode != 0:
        err_text = res.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg 合併多段影片失敗:\n{err_text}")

    return final_output_path


def prepare_reference_images_for_comfy(
    resolved_refs: List[Tuple[str, Path]],
    comfy_input_dir: Path,
    dry_run: bool = False,
) -> List[str]:
    """
    將解析到的人物定裝參考圖複製至 ComfyUI/input/，回傳供 LoadImage 節點使用的檔名清單。
    """
    comfy_filenames: List[str] = []
    if not resolved_refs:
        return comfy_filenames

    if not dry_run:
        comfy_input_dir.mkdir(parents=True, exist_ok=True)

    for idx, (_label, src_path) in enumerate(resolved_refs, start=1):
        safe_name = f"_ref_{idx}_{src_path.name}"
        dest_path = comfy_input_dir / safe_name
        if not dry_run and src_path.resolve() != dest_path.resolve():
            shutil.copy2(str(src_path), str(dest_path))
        comfy_filenames.append(safe_name)

    return comfy_filenames


def main():
    parser = argparse.ArgumentParser(
        description="讀取 prompt.txt 與人物定裝參考圖 (person1.jpg / person2.jpg)，使用 MiniMax H3 Ref2VA HyperFlow Flash + SeedVR2 產生有聲影片"
    )
    parser.add_argument("--resolution", type=str, default=None, help="指定畫面解析度，如 1024x768 或 1920x1080")
    parser.add_argument("--duration", type=float, default=None, help="指定影片總秒數，預設 15 秒")
    parser.add_argument("--steps", type=int, default=6, choices=[4, 6, 8], help="HyperFlow 採樣步數 (預設 6 步)")
    parser.add_argument("--seed", type=int, default=None, help="隨機種子 (預設隨機產生)")
    parser.add_argument(
        "--ref-image-size",
        type=str,
        default="match",
        choices=["match", "max"],
        help="參考圖輸入尺寸策略：match (預設，符合輸出面積較快) 或 max (2048px 短邊，臉部特徵最精準但較慢)",
    )
    parser.add_argument("--prompt-file", type=str, default=str(PROMPT_FILE), help="提示詞檔案路徑 (預設 ./prompt.txt)")
    parser.add_argument("--yes", "-y", action="store_true", help="使用預設參數不互動詢問")
    parser.add_argument("--dry-run", action="store_true", help="僅產生並驗證工作流 JSON，不實際執行推論")
    args = parser.parse_args()

    prompt_path = Path(args.prompt_file).resolve()
    raw_prompt_text = read_prompt_file(prompt_path)

    # 解析固定參考圖 (person1.jpg, person2.jpg) 與提示詞中的 <檔名> 標籤
    ref_search_dirs = [prompt_path.parent, ROOT_DIR, ROOT_DIR / "refs", COMFY_DIR / "input"]
    # 去除重複搜尋目錄同時保留順序
    dedup_search_dirs: List[Path] = []
    for d in ref_search_dirs:
        rd = d.resolve()
        if rd not in dedup_search_dirs:
            dedup_search_dirs.append(rd)

    prompt_text, resolved_refs, missing_tags = parse_and_resolve_references(
        prompt_text=raw_prompt_text,
        search_dirs=dedup_search_dirs,
        fixed_files=DEFAULT_REF_IMAGE_FILES,
    )
    comfy_ref_filenames = prepare_reference_images_for_comfy(
        resolved_refs=resolved_refs,
        comfy_input_dir=COMFY_DIR / "input",
        dry_run=args.dry_run,
    )

    print("========================================================")
    print(f"📄 讀取提示詞檔案: {prompt_path}")
    print(f"💬 原始提示詞:\n   {raw_prompt_text}")
    if resolved_refs:
        print(f"🖼️  已載入人物定裝參考圖 ({len(resolved_refs)} 張, ref_image_size={args.ref_image_size}):")
        for idx, (label, img_path) in enumerate(resolved_refs, start=1):
            print(f"   - <Picture {idx}> <- {label} ({img_path})")
    else:
        print("ℹ️  未偵測到 person1.jpg / person2.jpg 或其他參考圖，將以純文字 Ref2VA 模式生成")
    if missing_tags:
        print(f"⚠️  以下提示詞參考圖找不到對應檔案，已自動忽略跳過: {', '.join(f'<{t}>' for t in missing_tags)}")
    if prompt_text != raw_prompt_text:
        print(f"✨ 轉換後提示詞 (送入 MiniMaxH3ReferenceToVideo):\n   {prompt_text}")
    print("========================================================")

    target_w, target_h, total_seconds, steps = ask_user_parameters(
        cli_resolution=args.resolution,
        cli_duration=args.duration,
        cli_steps=args.steps,
        non_interactive=args.yes,
    )

    need_seedvr2_upscale = (target_w > DEFAULT_WIDTH) or (target_h > DEFAULT_HEIGHT)
    if need_seedvr2_upscale:
        # 維持目標長寬比，將基礎生成解析度控制在約 1024x768 像素量級，再交由 SeedVR2 放大
        aspect = target_w / target_h
        base_pixels = DEFAULT_WIDTH * DEFAULT_HEIGHT
        gen_h = int(round(math.sqrt(base_pixels / aspect) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE
        gen_w = int(round((gen_h * aspect) / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE
        target_short_edge = min(target_w, target_h)
        target_max_edge = max(target_w, target_h)
        print(
            f"🔍 目標解析度 {target_w}x{target_h} 大於預設 {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}：\n"
            f"   -> 將先以基礎解析度 {gen_w}x{gen_h} 生成，並自動啟用 SeedVR2 (3B) 放大至 {target_w}x{target_h}！"
        )
    else:
        gen_w, gen_h = target_w, target_h
        target_short_edge = min(target_w, target_h)
        target_max_edge = max(target_w, target_h)
        print(f"📐 使用原生解析度生成: {gen_w}x{gen_h} (無需額外放大)")

    # 計算分段數量 (每段最長 SEGMENT_MAX_SECONDS = 5 秒，若影片很長則把最後一格作為開頭錨點並持續注入人物參考圖)
    num_segments = max(1, int(math.ceil(total_seconds / SEGMENT_MAX_SECONDS)))
    segment_durations = []
    rem = total_seconds
    for _ in range(num_segments):
        seg_sec = min(SEGMENT_MAX_SECONDS, rem)
        # 若剩餘秒數太短 (< 2 秒)，確保至少產生 2 秒以維持影音品質
        seg_sec = max(2.0, seg_sec)
        segment_durations.append(seg_sec)
        rem = max(0.0, rem - SEGMENT_MAX_SECONDS)

    print(
        f"⏱️  總目標秒數: {total_seconds:.1f} 秒 -> 共分為 {num_segments} 段生成 "
        f"({', '.join(f'{d:.1f}s' for d in segment_durations)})"
    )
    if num_segments > 1:
        print("🔗 長影片接續模式已啟用：每段將錨定上一段最後一格，並持續注入人物定裝參考圖保持角色一致！")

    base_seed = args.seed if args.seed is not None else random.randint(1, 2**31 - 1)
    WORKFLOWS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for idx, seg_sec in enumerate(segment_durations, start=1):
            frames = seconds_to_h3_frames(seg_sec)
            wf = build_hyperflow_workflow(
                prompt=prompt_text,
                width=gen_w,
                height=gen_h,
                length_frames=frames,
                steps=steps,
                seed=base_seed + idx - 1,
                ref_images=comfy_ref_filenames,
                ref_image_size=args.ref_image_size,
                first_frame_image=f"continuation_frame_{idx-1}.png" if idx > 1 else None,
                upscale_with_seedvr2=need_seedvr2_upscale,
                target_short_edge=target_short_edge,
                target_max_edge=target_max_edge,
                filename_prefix=f"hyperflow/seg_{idx:02d}",
            )
            wf_path = WORKFLOWS_TMP_DIR / f"workflow_seg_{idx:02d}.json"
            wf_path.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ [Dry-Run] 已產生第 {idx} 段工作流 JSON: {wf_path}")
        return

    # 1. 檢查並確保模型已下載，並套用 Apple Silicon 無損加速補丁
    patch_hyperflow_gguf(COMFY_DIR)
    print("\n🔎 正在檢查模型檔案完整性...")
    if not ensure_models(COMFY_DIR, include_seedvr2=need_seedvr2_upscale):
        print("❌ 模型準備未完成，請先執行 ./setup.sh 或檢查網路連線。", file=sys.stderr)
        sys.exit(1)

    # 2. 啟動本地 ComfyUI 伺服器
    server_proc = start_comfy_server_via_cli()

    try:
        segment_video_paths: List[Path] = []
        last_frame_filename: Optional[str] = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        overall_pbar = tqdm(
            total=num_segments,
            desc="📊 整體影片生成總進度",
            position=2,
            leave=True,
        )

        for idx, seg_sec in enumerate(segment_durations, start=1):
            frames = seconds_to_h3_frames(seg_sec)
            ref_info = f"{len(comfy_ref_filenames)} 張參考圖" if comfy_ref_filenames else "無參考圖"
            mode_label = (
                f"Ref2VA 首段生成 ({ref_info})"
                if idx == 1
                else f"Ref2VA 接續生成 ({ref_info}, 首格錨點: {last_frame_filename})"
            )
            print(f"\n▶️  開始執行第 {idx}/{num_segments} 段 [{mode_label}] ({seg_sec:.1f} 秒 / {frames} 影格, {steps} 步)...")

            wf = build_hyperflow_workflow(
                prompt=prompt_text,
                width=gen_w,
                height=gen_h,
                length_frames=frames,
                steps=steps,
                seed=base_seed + idx - 1,
                ref_images=comfy_ref_filenames,
                ref_image_size=args.ref_image_size,
                first_frame_image=last_frame_filename,
                upscale_with_seedvr2=need_seedvr2_upscale,
                target_short_edge=target_short_edge,
                target_max_edge=target_max_edge,
                filename_prefix=f"hyperflow/{timestamp}_seg_{idx:02d}",
            )

            wf_path = WORKFLOWS_TMP_DIR / f"{timestamp}_seg_{idx:02d}.json"
            wf_path.write_text(json.dumps(wf, indent=2, ensure_ascii=False), encoding="utf-8")

            seg_video = execute_workflow_with_progress(
                workflow=wf,
                segment_idx=idx,
                total_segments=num_segments,
            )
            segment_video_paths.append(seg_video)
            overall_pbar.update(1)

            # 若還有下一段，擷取本段最後一格存入 ComfyUI/input/
            if idx < num_segments:
                last_frame_filename = f"{timestamp}_last_frame_{idx:02d}.png"
                input_frame_path = COMFY_DIR / "input" / last_frame_filename
                extract_last_frame(seg_video, input_frame_path)
                print(f"🖼️  已擷取第 {idx} 段最後一格影格 -> {input_frame_path.name}")

        overall_pbar.close()

        # 3. 合併所有分段並輸出至 outputs/
        final_filename = f"minimax_h3_ref2va_{target_w}x{target_h}_{int(total_seconds)}s_{timestamp}.mp4"
        final_output_path = OUTPUTS_DIR / final_filename
        print(f"\n🎞️  正在合成最終有聲影片至 {final_output_path}...")
        concatenate_segments(segment_video_paths, final_output_path)

        size_mb = final_output_path.stat().st_size / (1024 * 1024)
        print("\n========================================================")
        print(f"🎉 影片產生完成！")
        print(f"📂 輸出檔案: {final_output_path} ({size_mb:.2f} MB)")
        print("========================================================")

    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.terminate()


if __name__ == "__main__":
    main()
