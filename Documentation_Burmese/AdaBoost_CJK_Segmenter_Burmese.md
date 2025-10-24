# AdaBoost CJK Segmenter ကို မြန်မာဘာသာအတွက် အသုံးပြုခြင်း

ဤစာတမ်းတွင် `adaboost_cjk_segmenter` ကို မြန်မာဘာသာစကားအတွက် မည်သို့အသုံးပြုနိုင်ကြောင်း၊ မည်သို့ dataset များပြင်ဆင်ရမည်ဖြစ်ကြောင်း၊ နှင့် model ကို မည်သို့ train ရမည်ဖြစ်ကြောင်း အသေးစိတ်ရှင်းလင်းဖော်ပြပါမည်။

## AdaBoost Model အကြောင်း

AdaBoost (Adaptive Boosting) သည် machine learning algorithm တစ်ခုဖြစ်ပြီး classification ပြဿနာများကို ဖြေရှင်းရန်အတွက် အသုံးပြုပါသည်။ ၎င်းသည် "weak learners" (ခန့်မှန်းချက် အနည်းငယ်သာ မှန်သော model ငယ်များ) အများအပြားကိုပေါင်းစပ်ပြီး "strong learner" (ခန့်မှန်းချက် ပိုမိုတိကျသော model ကြီးတစ်ခု) ကို တည်ဆောက်ပါသည်။

Word segmentation တွင် AdaBoost ကို စာလုံးတစ်လုံးစီ၏ နောက်တွင် word boundary (စကားလုံးအဆုံး) ရှိ၊ မရှိ ဆုံးဖြတ်ရန်အတွက် အသုံးပြုပါသည်။ AdaBoost ၏ အဓိက အလုပ်လုပ်ပုံမှာ-

1.  **Weak Learner များစွာကို Train ခြင်း:** Model သည် training data ကို အသုံးပြု၍ weak learner အများအပြားကို အဆင့်ဆင့် train ပါသည်။ weak learner တစ်ခုချင်းစီသည် data ၏ feature အချို့ကိုသာ အာရုံစိုက်ပြီး ရိုးရှင်းသော ဆုံးဖြတ်ချက်များကို ချမှတ်ပါသည်။
2.  **မှားယွင်းမှုများကို အာရုံစိုက်ခြင်း:** weak learner တစ်ခု train ပြီးတိုင်း၊ model သည် ၎င်း learner မှားယွင်းစွာခွဲခြားခဲ့သော data point များကို ပိုမိုအာရုံစိုက်ပါသည်။ နောက်ထပ် train မည့် weak learner သည် ထိုမှားယွင်းမှုများကို ပြင်ဆင်ရန် ကြိုးစားရမည်ဖြစ်ပါသည်။
3.  **Learner များကို ပေါင်းစပ်ခြင်း:** နောက်ဆုံးတွင်၊ AdaBoost သည် weak learner အားလုံး၏ ခန့်မှန်းချက်များကို ပေါင်းစပ်ပြီး၊ learner တစ်ခုချင်းစီ၏ တိကျမှုအပေါ်မူတည်၍ အလေးချိန် (weight) များပေးကာ နောက်ဆုံး ဆုံးဖြတ်ချက်ကို ချမှတ်ပါသည်။

## Model ၏ အလုပ်လုပ်ပုံ

`adaboost_cjk_segmenter` သည် တရုတ်ဘာသာစကားအတွက် တည်ဆောက်ထားပြီး တရုတ်စာလုံးများ၏ "radicals" (စာလုံးများ၏ အဓိပ္ပာယ်ကို ကိုယ်စားပြုသော အစိတ်အပိုင်းများ) ကို အဓိက feature အဖြစ်အသုံးပြုပါသည်။ သို့သော် မြန်မာစာတွင် radicals များမရှိသောကြောင့် ဤ model ကို တိုက်ရိုက်အသုံးပြု၍မရပါ။

မြန်မာဘာသာအတွက် ဤ model ကို အသုံးပြုလိုပါက၊ feature extraction ပိုင်းကို ပြန်လည်ပြင်ဆင်ရေးသားရန် လိုအပ်မည်ဖြစ်ပါသည်။ တရုတ်စာလုံး radicals များအစား၊ မြန်မာဘာသာစကား၏ grapheme clusters (ဥပမာ "က", "ကျ", "ချောင်း") သို့မဟုတ် အခြားသော linguistic features များကို အသုံးပြုရမည်ဖြစ်ပါသည်။

## Dataset ပြင်ဆင်ခြင်း

Model ကို train ရန်အတွက်၊ "segmented" လုပ်ထားသော dataset တစ်ခုလိုအပ်ပါသည်။ ၎င်းသည် စကားလုံးများကို `|` ကဲ့သို့သော သင်္ကေတများဖြင့် ပိုင်းခြားထားသော စာသားများဖြစ်ပါသည်။

ဥပမာ:

```
|မြန်မာစာ|သည်|အလွန်|လှပသော|ဘာသာစကား|တစ်ခု|ဖြစ်သည်|။|
```

### Dataset ကို Segmentation လုပ်ခြင်း

Dataset ကို segmentation လုပ်ရန်အတွက် နည်းလမ်းနှစ်ခုရှိပါသည်။

1.  **Manually Segmented Data:** လူကိုယ်တိုင် စကားလုံးများကို ပိုင်းခြားထားသော dataset များသည် အတိကျဆုံးဖြစ်ပြီး model ၏ performance ကို အကောင်းဆုံးဖြစ်စေပါသည်။
2.  **Pseudo-segmented Data:** အကယ်၍ manually segmented data မရှိပါက၊ ICU (International Components for Unicode) ကဲ့သို့သော လက်ရှိ segmentation tool များကိုအသုံးပြု၍ pseudo-segmented data များကို ဖန်တီးနိုင်ပါသည်။

## Model Train ရန် ပြင်ဆင်ခြင်း

1.  **Feature Extractor ကို ပြင်ဆင်ခြင်း:** `parser.py` file ထဲရှိ `get_radical` function ကို မြန်မာဘာသာအတွက် အသုံးဝင်သော feature များကို ထုတ်ပေးနိုင်သော function အသစ်ဖြင့် အစားထိုးရန်လိုအပ်ပါသည်။ မြန်မာဘာသာအတွက် အောက်ပါ feature များကို အသုံးပြုနိုင်ပါသည်-
    *   **Grapheme Clusters:** လက်ရှိ စာလုံးနှင့် ၎င်း၏ အရှေ့၊ အနောက်ရှိ grapheme cluster များ။
    *   **Character Types:** စာလုံးသည် ဗျည်း၊ သရ၊ အသတ်၊ သို့မဟုတ် အခြား သင်္ကေတတစ်ခုခု ဟုတ်မဟုတ်။
    *   **N-grams:** Unigrams (စာလုံးတစ်လုံးချင်း)၊ bigrams (စာလုံးနှစ်လုံးတွဲ)၊ နှင့် trigrams (စာလုံးသုံးလုံးတွဲ) များ။
2.  **Training Script ကို ရေးသားခြင်း:** AdaBoost model ကို train ရန်အတွက် training script အသစ်တစ်ခုကို ရေးသားရန်လိုအပ်ပါသည်။ ဤ script သည် segmented data ကိုဖတ်ပြီး၊ feature များကိုထုတ်ယူကာ၊ model ကို train ရမည်ဖြစ်ပါသည်။ Scikit-learn ကဲ့သို့သော library များကို အသုံးပြု၍ AdaBoostClassifier ကို အလွယ်တကူ train နိုင်ပါသည်။

## Model Train ခြင်း အဆင့်များ

1.  **Dataset ကို ပြင်ဆင်ပါ:** အထက်တွင်ဖော်ပြထားသည့်အတိုင်း segmented dataset ကို ပြင်ဆင်ပါ။
2.  **Feature Extractor ကို ရေးသားပါ:** မြန်မာဘာသာအတွက် feature extractor ကို `parser.py` တွင် ရေးသားပါ။
3.  **Training Script ကို Run ပါ:** Training script ကို run ၍ model ကို train ပါ။ ပြီးစီးပါက `model.json` file အသစ်တစ်ခု ရရှိပါမည်။
4.  **Model ကို အသုံးပြုခြင်း:** ရရှိလာသော `model.json` file ကို `AdaBoostSegmenter` class တွင် load လုပ်ပြီး မြန်မာစာသားများကို segment လုပ်ရန် အသုံးပြုနိုင်ပါသည်။

## နမူနာ Code (မြန်မာဘာသာအတွက် ပြင်ဆင်ထားသော)

```python
# parser.py (ပြင်ဆင်ရန်လိုအပ်သည်)
def get_burmese_features(char1, char2):
    # ဤနေရာတွင် မြန်မာဘာသာအတွက် feature များကို ထုတ်ယူရန် code ရေးသားရန်
    # ဥပမာ: grapheme clusters, character types, etc.
    features = {}
    # ...
    return features

class AdaBoostSegmenter:
    def __init__(self, model):
        self.model = model

    def predict(self, sentence):
        if sentence == '':
            return []
        chunks = [sentence[0]]
        base_score = -sum(sum(g.values()) for g in self.model.values()) * 0.5

        for i in range(1, len(sentence)):
            score = base_score
            L = len(chunks[-1])
            score += 32**L

            # မြန်မာဘာသာအတွက် feature များကို အသုံးပြုခြင်း
            features = get_burmese_features(sentence[i-1], sentence[i])
            for feature_name, feature_value in features.items():
                score += self.model.get(feature_name, {}).get(feature_value, 0)

            if score > 0:
                chunks.append(sentence[i])
            else:
                chunks[-1] += sentence[i]
        return chunks
```

ဤ documentation သည် `adaboost_cjk_segmenter` ကို မြန်မာဘာသာအတွက် အသုံးပြုရန် လမ်းညွှန်ချက်များ ပေးထားခြင်းဖြစ်ပါသည်။ လက်တွေ့တွင်၊ feature engineering နှင့် training process များသည် ပိုမိုရှုပ်ထွေးနိုင်ပြီး၊ မြန်မာဘာသာစကား၏ သဘောသဘာဝကို နားလည်ရန် အရေးကြီးပါသည်။
