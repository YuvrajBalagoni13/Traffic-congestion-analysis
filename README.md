# TRAFFIC CONGESTION ANALYSIS - BARBADOS TRAFFIC ANALYSIS CHALLENGE

# Problem Statement
## Overview

In this challenge, the task was to predict traffic congestion using machine learning, with the aim of recognizing the root causes of traffic in a specific roundabout in Barbados.
We were provided with four streams of video data, labelled with congestion rating for the entrance & exit timestamps. The models aim was to predict traffic congestion five minutes into the future. Ultimately the challenge was focused on identifying the root causes of increased time spent in the roundabout by developing features from unstructured video data.

## Evaluation

The challenge focused on 2 metrics -
1. Macro-F1 (70%) : measuring how well model performs across all 4 congestion classes, treating each class equally.
2. Accuracy (30%) : measuring the overall percentage of correct predictions across all samples.

# Solution Summary 

So my approach was to generate features from the unstructured video data using a lightweight detection & tracking model (YOLO26-nano & bytetrack), & using those features to predict the congestion in traffic 5 minutes into the future.

## Data

- **Raw training data** : 16,076 rows from 4 traffic cameras. (16,076 video data with each data being 1 minute long).
- **Training Data for fine-tuning YOLO model** : Extracted frames from -
	- Training samples - 16k images
	- Validation samples - 5k images
- **Training data for XGBoost model** : 
	- 2965 rows (due to resource constraints + intentional reduction to handle imbalance & removing rows with missing past or future timesteps).

## Detailed Approach

- Used YOLO26-nano model + ByteTrack to detect & track vehicles (fine tuned it using dataset generated from rfdetr model).
- Now based on these detections created features
	- for features I added regions in the entry & exit for every camera for all the entries & exits of the roundabout.
	- Now based of these regions the features are generated
	- Also there is line which is used to track the count of the vehicles already in the roundabout
    <a href="https://github.com/YuvrajBalagoni13/Traffic-congestion-analysis/blob/main/imgs/normanniles1.png">
  <img width="100%" alt="Integration" src="https://raw.githubusercontent.com/YuvrajBalagoni13/Traffic-congestion-analysis/main/imgs/normanniles1.png" />
</a>
<a href="https://github.com/YuvrajBalagoni13/Traffic-congestion-analysis/blob/main/imgs/normanniles2.png">
  <img width="100%" alt="Integration" src="https://raw.githubusercontent.com/YuvrajBalagoni13/Traffic-congestion-analysis/main/imgs/normanniles2.png" />
</a>
	- ### Features for every 1 min video-
		1. Number of vehicles passing through Entry
		2. Number of vehicles passing through Entry but already in roundabout
		3. Number of vehicles passing through Exit
		4. Average time spent in entry
		5. Average time spent in exit
		6. Average time spent in entry by vehicles already in roundabout
		7. Average idle time for vehicles in entry
		8. Average idle time for vehicles in exit 
		9. Average idle time for vehicles already in roundabout
		10. Buckets of idle time for vehicles in entry - (0-2, 2-4, 4-6, 6-8, >8) in seconds
		11. Buckets of idle time for vehicles in exit - (0-2, 2-4, 4-6, 6-8, >8) in seconds
		12. Buckets of time spent by vehicles in entry - (0-2, 2-4, 4-6, 6-8, >8) in seconds
		13. Buckets of time spent by vehicles in entry - (0-2, 2-4, 4-6, 6-8, >8) in seconds
		14. Average of the idle time bucket for last 10 minutes
		15. previous 10 minutes features for congestion enter rating, congestion exit rating, signaling, number of vehicles entry, number of vehicles exit, number of vehicles in roundabout, idle entry times, time spent in entry.
		    
		    ### Features that were already given in train.csv of the challenge
		16. Camera Info (which entry/exit the camera is for) exg. normanniles1, normanniles2, etc.
		17. Signaling 
		18. hour, day time (night, morning, etc), minute, day of week
		19. hour sin & hour cos
		20. is rush hour, is morning rush & is evening rush

		In total we had a total of 165 features
- Now another major issue in this challenge was the imbalance in the dataset
<a href="https://github.com/YuvrajBalagoni13/Traffic-congestion-analysis/blob/main/imgs/imbalance_entire_dataset.png">
  <img width="100%" alt="Integration" src="https://raw.githubusercontent.com/YuvrajBalagoni13/Traffic-congestion-analysis/main/imgs/imbalance_entire_dataset.png" />
</a>
- For this issue i just downsampled the majority class (free flowing) 
  IMP - Due to resource constraints, it was not possible to process the entire 16k videos as each video was taking about 10-15 secs on T4 GPU (on google Colab) which would be about 67 hrs. Instead I did around 3k videos (I did downsampling before processing the videos for efficiency).
  In downsampling I removed about 40% of the free flowing samples which had free flowing for the next 5 minutes. That means I samples is considered for getting removed if its 5 mins future are also free flowing.
- So the final dataset length remaining was 2372 samples with video features & ---- samples without video features.
- This is after doing all the pre processing including removing samples with previous & future NaN values due to feature engineering.
- Later I did a 5-fold Stratified Cross-Validation pipeline using LightGBM, adopting a "direct" multi-target forecasting approach by training a completely independent classification model for each future time step.
- Within each fold, there was early stopping based on a custom macro F1 metric to prevent overfitting, and finally aggregated the predictions across all individual models to compute the overall global performance.
## Results & Model Performance

The predictive modeling was evaluated using a **5-fold Stratified Cross-Validation** approach with a LightGBM Classifier. 

Because the traffic congestion dataset is highly imbalanced (with the "free-flowing" class making up the vast majority of the data), **Macro F1-Score** was chosen as the primary evaluation metric over raw Accuracy. This ensures the model is penalized for missing rare, severe traffic events.

I experimented with two distinct data balancing strategies:
1. **With Random Oversampling:** Physically duplicating minority class samples to reach 70% of the majority class volume.
2. **Without Oversampling:** Relying entirely on LightGBM's native `class_weight='balanced'` parameter to apply mathematical penalties.

### Cross-Validation Metrics

| Fold | With Oversampling (Acc) | With Oversampling (F1) | Without Oversampling (Acc) | Without Oversampling (F1) |
| :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.5157 | 0.4205 | 0.4823 | **0.4411** |
| **Fold 2** | 0.5325 | 0.4257 | 0.4958 | **0.4583** |
| **Fold 3** | 0.5572 | 0.4501 | 0.5201 | **0.4787** |
| **Fold 4** | 0.5322 | 0.4391 | 0.4951 | **0.4681** |
| **Fold 5** | 0.5282 | 0.4234 | 0.4968 | **0.4597** |
| **Final Mean** | 0.5332 | 0.4318 | 0.4980 | **0.4612** |

### Key Takeaways

* **The Best Approach:** The **"Without Oversampling"** strategy got the best performance, achieving a final **Macro F1-Score of 0.4612**.
* **The Accuracy vs. F1 Trade-off:** While random oversampling artificially increased the overall Accuracy (~53%), it dragged down the F1-Macro score (~43%). This occurred because exact data duplication caused the tree-based models to overfit on memorized points, reducing the algorithm's built-in class penalties.
* **Conclusion:** By relying strictly on the downsampled dataset and LightGBM's `class_weight='balanced'` parameter, the model performed better.