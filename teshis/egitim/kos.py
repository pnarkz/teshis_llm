"""Start a controlled Ultralytics training run from a versioned YAML."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .kayit import write_run_manifest
from .protokol import egitim_kwargs, egitim_kwargs_e


def devam_et(run_dir: Path, onayla: bool = False) -> Path:
    """Yarida kalmis bir kosuyu son checkpoint'ten surdurur. DIKKAT: guvenilir degil.

    !!! OLCULEN SORUN !!!
    Bu ozellik v00n kosusunda denendi ve egitimi BOZDU. Kesinti oncesi ve
    sonrasi (ayni kosu, ayni protokol, LR duzgun azaliyor):

        epoch 23 (kesintiden once): mAP50 0.8136, cls_loss 0.7477
        epoch 24 (devam sonrasi)  : mAP50 0.2849, cls_loss 3.1210
        epoch 25                  : mAP50 0.3473, cls_loss 1.8099

    cls_loss dort katina cikti ve kismi toparlanmaya ragmen kosu eski
    seviyesine donemedi. Kopus tam olarak devam sinirinda; Ultralytics'in
    optimizer/EMA durumunu tam geri yukleyememesine isaret ediyor.

    Bu yuzden devam etmek, bilimsel karsilastirmaya girecek bir kosu icin
    KULLANILMAMALIDIR: elde edilen model, bastan kosulmus bir modelle ayni
    protokolun urunu sayilamaz. Kesinti olursa dogru davranis, kaydedilmis
    best.pt'yi kullanmak veya kosuyu bastan baslatmaktir.

    Ozellik silinmedi cunku kesif/duman testi gibi karsilastirmaya girmeyen
    kosularda hala ise yarayabilir; ancak bilincli onay ister.
    """
    from ultralytics import YOLO

    son = run_dir.resolve() / "weights/last.pt"
    if not son.is_file():
        raise FileNotFoundError(
            f"Devam edilecek checkpoint yok: {son}\n"
            "Kosu hic epoch tamamlamamis olabilir; bastan baslatin."
        )
    if not onayla:
        raise RuntimeError(
            "Devam etmek egitimi bozabilir (bkz. devam_et docstring: v00n kosusunda "
            "cls_loss 0.75 -> 3.12 sicradi ve kosu toparlanamadi).\n"
            "Bilimsel karsilastirmaya girecek bir kosuda KULLANMAYIN; bunun yerine "
            "kaydedilmis best.pt'yi kullanin veya kosuyu bastan baslatin.\n"
            "Yine de devam etmek istiyorsaniz --devam-onayla ekleyin."
        )
    print(f"UYARI: devam ediliyor, egitim bozulabilir -> {son}")
    YOLO(str(son)).train(resume=True)
    return run_dir


def run_training(
    model_path: Path,
    data_path: Path,
    output_root: Path,
    scenario: str,
    epochs: int,
    batch: int,
    imgsz: int,
    seed: int,
    device: int | str,
    workers: int,
    e_senaryo: str | None = None,
) -> Path:
    """Train without touching source data or the selected source model.

    `e_senaryo` verilirse ortak protokol, o E senaryosunun YAML'da BEYAN
    EDILMIS sapmalariyla birlestirilir (bkz. senaryolar/egitim_protokolu.yaml
    -> e_serisi). D serisi kosulari bu parametreyi kullanmaz; onlarda tek
    degisken veri surumudur.
    """
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Bu komut ultralytics ve torch kurulu yerel/GPU ortaminda calisir.") from error

    model_path = model_path.resolve()
    data_path = data_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bulunamadi: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Data YAML bulunamadi: {data_path}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU istendi fakat CUDA kullanilamiyor. --device cpu ile bilincli baslatin.")

    protokol_kwargs = egitim_kwargs_e(e_senaryo) if e_senaryo else egitim_kwargs()
    sapmalar = {k: v for k, v in protokol_kwargs.items() if egitim_kwargs().get(k) != v}
    if sapmalar:
        print(f"UYARI: {e_senaryo} ortak protokolden KASITLI olarak sapiyor: {sapmalar}")

    run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}_{scenario}_{seed}"
    run_dir = output_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    write_run_manifest(
        run_dir,
        {
            "scenario": scenario,
            "model": str(model_path),
            "data": str(data_path),
            "device": str(device),
            "seed": seed,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "test_evaluated_during_training": False,
            # Kosunun kendisi hangi protokolle uretildigini tasisin: rapor
            # okunurken YAML'in o gunku haline geri donmek gerekmesin.
            "e_senaryo": e_senaryo,
            "protokol_sapmalari": sapmalar,
        },
    )

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        device=device,
        workers=workers,
        seed=seed,
        project=str(output_root.resolve()),
        name=run_name,
        exist_ok=True,
        # D1 tanı kosusunda labels.jpg uretimi Windows'ta takilabildigi icin kapali.
        plots=False,
        val=True,
        pretrained=str(model_path),
        **protokol_kwargs,
    )
    print(json.dumps({"run_dir": str(run_dir), "weights": str(run_dir / "weights")}, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Versioned YOLO training run")
    parser.add_argument("--model", default="main_model.pt")
    parser.add_argument("--data", default="veri_surumleri/v01_d1_sinif_yetersizligi/data.yaml")
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--scenario", default="D1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0", help="GPU id, or cpu")
    parser.add_argument("--workers", type=int, default=0, help="Windows yerel kosuda 0 daha kararlidir")
    # Secenekler PROTOKOLDEN turetilir; elle liste tutmak YAML ile ayrisir.
    # E3b eklendiginde tam bunun oldugu gorildu: senaryo protokolde tanimliydi
    # ama CLI "invalid choice: 'E3b'" diyerek kosuyu reddetti.
    from .protokol import yukle

    e_kodlari = sorted(yukle().get("e_serisi", {}))
    parser.add_argument(
        "--e-senaryo",
        choices=e_kodlari,
        help="E serisi egitim arizasi: ortak protokole YAML'da beyan edilmis "
             "sapmalari uygular. Egitim gerektirmeyen senaryolar (orn. E4) "
             "burada secilse de bir sey degistirmez.",
    )
    parser.add_argument(
        "--devam", type=Path, default=None,
        help="Yarida kalmis bir kosuyu surdur. DIKKAT: olculen bir sorun var, "
             "egitimi bozabilir; karsilastirmaya girecek kosularda kullanmayin.",
    )
    parser.add_argument(
        "--devam-onayla", action="store_true",
        help="--devam riskini bilerek kabul et (bkz. kos.py::devam_et).",
    )
    args = parser.parse_args()

    if args.devam:
        devam_et(args.devam, onayla=args.devam_onayla)
        return
    device: int | str = "cpu" if args.device.lower() == "cpu" else int(args.device)
    run_training(
        Path(args.model), Path(args.data), Path(args.output_root), args.scenario,
        args.epochs, args.batch, args.imgsz, args.seed, device, args.workers,
        e_senaryo=args.e_senaryo,
    )


if __name__ == "__main__":
    main()
