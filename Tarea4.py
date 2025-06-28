from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pandas as pd
from totalsegmentator.python_api import totalsegmentator


def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()
    return 2 * inter / union if union > 0 else 1.0


def main():
    dataset_path = Path("spine_seg/home/shared_data/kaggle-spine-seg/spine_segmentation_nnunet_v2")
    volumes_dir = dataset_path / "volumes"
    segmentations_dir = dataset_path / "segmentations"
    output_dir = Path("full_segmentation_outputs")
    output_dir.mkdir(exist_ok=True)

    # Etiquetas predicha vs ground truth
    pred_ids = [47, 43, 41]  # L4, T10, T8 según TotalSegmentator
    gt_ids = [4, 8, 10]      # L4, T10, T8 en dataset
    vertebra_names = ['L4', 'T10', 'T8']

    volume_paths = []
    for vol_path in sorted(volumes_dir.glob("*.nii")):
        seg_path = segmentations_dir / vol_path.name
        if seg_path.exists():
            volume_paths.append((vol_path, seg_path))

    print(f"✅ Total de volúmenes válidos: {len(volume_paths)}")

    dice_results = {name: [] for name in vertebra_names}
    dice_cases = {name: [] for name in vertebra_names}
    case_dice_scores = []

    for i, (vol_path, gt_path) in enumerate(volume_paths[:100]):
        case_id = vol_path.stem
        print(f"\n🔄 Procesando {case_id} ({i+1}/100)...")
        output_path = output_dir / f"ts_output_{case_id}.nii.gz"

        if not output_path.exists():
            totalsegmentator(
                input=str(vol_path),
                output=str(output_path),
                task="total",
                ml=True,
                fast=True,
                statistics=False,
                quiet=True
            )

        pred_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(output_path)))
        gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path)))

        per_case_scores = []
        for name, pred_id, gt_id in zip(vertebra_names, pred_ids, gt_ids):
            if pred_id in np.unique(pred_mask) and gt_id in np.unique(gt_mask):
                score = dice_score(pred_mask == pred_id, gt_mask == gt_id)
                dice_results[name].append(score)
                dice_cases[name].append((case_id, score))
                per_case_scores.append(score)
            else:
                per_case_scores.append(0.0)  # o np.nan si prefieres ignorarlas
                dice_results[name].append(0.0)

        avg_case_score = np.mean(per_case_scores)
        case_dice_scores.append((case_id, avg_case_score))


    # Crear lista sin ceros para estadísticas "reales"
    filtered_results = {
        name: [score for score in dice_results[name] if score > 0]
        for name in vertebra_names
    }

    # Mostrar resumen
    print("\n📊 Resumen por vértebra:")
    df = pd.DataFrame({
        "Vertebra": vertebra_names,
        "DICE Mean": [np.mean(filtered_results[v]) if filtered_results[v] else np.nan for v in vertebra_names],
        "DICE Std": [np.std(filtered_results[v]) if filtered_results[v] else np.nan for v in vertebra_names],
        "Samples (valid)": [len(filtered_results[v]) for v in vertebra_names],
        "Samples (total)": [len(dice_results[v]) for v in vertebra_names]
    })

    print(df.to_string(index=False))

    # Top 3 y Peor 3 casos
    case_dice_scores.sort(key=lambda x: x[1], reverse=True)
    print("\n📈 Mejores y peores casos por vértebra:")
    for name in vertebra_names:
        sorted_cases = sorted(dice_cases[name], key=lambda x: x[1], reverse=True)
        print(f"\n🔹 {name} (total: {len(sorted_cases)} casos):")
        print("Top 3:")
        for cid, score in sorted_cases[:3]:
            print(f" - {cid}: {score:.4f}")
        print("Bottom 3:")
        for cid, score in sorted_cases[-3:]:
            print(f" - {cid}: {score:.4f}")








    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    def guardar_imagen(vol_path, mask_path, case_id, vertebra_name, label_id, tag):
        volume = sitk.GetArrayFromImage(sitk.ReadImage(str(vol_path)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

        mask_bin = (mask == label_id).astype(np.uint8)
        if mask_bin.sum() == 0:
            print(f"⚠️ {case_id} - {vertebra_name} no presente, se omite visualización.")
            return

        z_idx = np.argmax(mask_bin.sum(axis=(1, 2)))  # slice con mayor presencia
        slice_vol = volume[z_idx][::-1, :]
        slice_mask = mask_bin[z_idx][::-1, :]

        cmap = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 0.5]])  # rojo translúcido
        plt.figure(figsize=(5, 5))
        plt.imshow(slice_vol, cmap="gray")
        plt.imshow(slice_mask, cmap=cmap)
        plt.axis("off")
        fname = f"example_{vertebra_name}_{tag}_{case_id}.png"
        plt.savefig(fname, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"✅ Guardada: {fname}")

    print("\n💾 Guardando ejemplos visuales...")

    for name, pred_id in zip(vertebra_names, pred_ids):
        sorted_cases = sorted(dice_cases[name], key=lambda x: x[1], reverse=True)
        mejores = sorted_cases[:1]
        peores = sorted_cases[-1:]

        for tag, case in zip(["best", "worst"], mejores + peores):
            case_id, _ = case
            vol_path = volumes_dir / f"{case_id}.nii"
            pred_path = output_dir / f"ts_output_{case_id}.nii.gz"
            if vol_path.exists() and pred_path.exists():
                guardar_imagen(vol_path, pred_path, case_id, name, pred_id, tag)
    



    # 🔚 Generar máscara filtrada para 3D Slicer (con solo L4, T10, T8)
    final_example = "case_0000"  # Puedes cambiar esto por el que quieras visualizar
    pred_path = output_dir / f"ts_output_{final_example}.nii.gz"
    if pred_path.exists():
        print(f"\n🧠 Generando máscara filtrada de {final_example} para 3D Slicer...")
        mask = sitk.ReadImage(str(pred_path))
        mask_arr = sitk.GetArrayFromImage(mask)
        
        filtered_mask = np.zeros_like(mask_arr)
        for idx in pred_ids:
            filtered_mask[mask_arr == idx] = idx

        filtered_img = sitk.GetImageFromArray(filtered_mask)
        filtered_img.CopyInformation(mask)
        
        out_name = f"{final_example}_filtered_L4_T10_T8.nii.gz"
        sitk.WriteImage(filtered_img, out_name)
        print(f"✅ Guardado: {out_name}")
    else:
        print(f"⚠️ No se encontró la predicción de {final_example}")


if __name__ == "__main__":
    main()
