# Evaluation Metrics for Brain Connectivity Classification
# Contains accuracy, AUC, precision/recall/F1, and sensitivity/specificity calculations

from math import log10
import torch
import numpy as np 
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch.nn.functional as F
from scipy.special import softmax
import scipy.stats

def PSNR(mse, peak=1.):
	# Peak Signal-to-Noise Ratio calculation
	return 10 * log10((peak ** 2) / mse)

class AverageMeter(object):
	# Utility class to track running averages during training
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(preds, labels):
	# Calculate classification accuracy from logits and true labels
	# Returns: (number_correct, accuracy_percentage)
	correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
	return np.sum(correct_prediction), np.mean(correct_prediction)

def auc(preds, labels, is_logit=True):
	# Calculate Area Under ROC Curve (AUC) for classification performance
	# Supports both multi-class and binary classification
	if is_logit:
		probs = softmax(preds, axis=1)  # Convert logits to probabilities
	else:
		probs = preds
	try:
		auc_out = roc_auc_score(labels, probs, multi_class='ovr')  # One-vs-rest for multiclass
	except:
		auc_out = 0  # Return 0 if AUC calculation fails
	return auc_out

def prf(preds, labels, is_logit=True):
	# Calculate Precision, Recall, and F1-score
	# Returns: [precision, recall, f1_score]
	pred_lab= np.argmax(preds, 1)  # Get predicted class labels
	p,r,f,s  = precision_recall_fscore_support(labels, pred_lab, average='macro', zero_division=0)
	return [p,r,f]


def numeric_score(pred, gt):
	# Calculate confusion matrix components (FP, FN, TP, TN)
	FP = float(np.sum((pred == 1) & (gt == 0)))  # False Positives
	FN = float(np.sum((pred == 0) & (gt == 1)))  # False Negatives
	TP = float(np.sum((pred == 1) & (gt == 1)))  # True Positives
	TN = float(np.sum((pred == 0) & (gt == 0)))  # True Negatives
	return FP, FN, TP, TN

def metrics(preds, labels):
	# Calculate sensitivity (recall) and specificity for binary classification
	# Returns: (sensitivity, specificity)
	preds = np.argmax(preds, 1)  # Convert logits to predicted labels
	FP, FN, TP, TN = numeric_score(preds, labels)
	sen = TP / (TP + FN + 1e-10)  # Sensitivity = TP / (TP + FN)
	spe = TN / (TN + FP + 1e-10)  # Specificity = TN / (TN + FP)

	return sen, spe

