import os
import cv2
from utils import load_model_and_labels, evaluate_known_images


def load_known_samples(root_dir):
    # root_dir/class_name/*.jpg
    samples = []
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"WARNING: unable to read '{img_path}'")
                    continue
                samples.append((img, class_name))
    return samples


def main():
    model, class_labels = load_model_and_labels()
    print("Loaded model and label mapping:")
    for idx, label in sorted(class_labels.items()):
        print(f"  {idx} -> {label}")

    known_dir = "known_samples"  # create this folder with per-class subfolders
    if not os.path.exists(known_dir):
        raise SystemExit(f"Directory '{known_dir}' not found. Create with class subfolders and sample images.")

    samples = load_known_samples(known_dir)
    if len(samples) == 0:
        raise SystemExit("No sample images were found in known_samples directory")

    report = evaluate_known_images(model, class_labels, samples)

    print("\nEVALUATION SUMMARY")
    print(f"  Total: {report['total']}")
    print(f"  Correct: {report['correct']}")
    print(f"  Accuracy: {report['accuracy']*100:.2f}%")
    print(f"  Uncertain count (<0.6): {report['uncertain']}")
    print(f"  Status: {report['status']}")
    if report.get('recommendation'):
        print(f"  Recommendation: {report['recommendation']}")

    print("\nDetailed records")
    for r in report.get('results', []):
        print(r)


if __name__ == '__main__':
    main()