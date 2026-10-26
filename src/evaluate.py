import torch
import logging

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    ytrues = []
    ypreds = []
    yprobs = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            ytrues.append(labels)
            ypreds.append(outputs)
            yprobs.append(probs)
            
    return ytrues, ypreds, yprobs, running_loss / len(loader)


def compute_metrics(ytrues, ypreds, yprobs, class_names, threshold=0.5):
    
    report = classification_report(ytrues, ypreds, target_names=class_names)
    print(report)
    logging.info("Classification Report:\n%s", report)

    auc = roc_auc_score(ytrues, yprobs[:, 1])
    print("ROC AUC Score:", auc)
    logging.info("ROC AUC Score: %s", auc)

    conf_matrix = confusion_matrix(ytrues, ypreds)
    print("Confusion Matrix:")
    print(conf_matrix)
    logging.info("Confusion Matrix:\n%s", conf_matrix)

    roc_curve_vals = roc_curve(ytrues, yprobs[:, 1])
    fpr, tpr, thresholds = roc_curve_vals
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC_Curve_' + '_'.join(class_names) + '.png')
    plt.show()

