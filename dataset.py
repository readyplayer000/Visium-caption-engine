import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def clean_text(text):
    """
    Cleans text by lowercasing, removing punctuation, numbers, and excess whitespaces.
    """
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_captions(data_dir):
    """
    Loads captions from data directory. Supports both Kaggle 'captions.txt' format
    and standard 'visium_token.txt' format.
    
    Returns a dictionary mapping image_id to a list of cleaned captions.
    """
    captions_dict = {}
    
    # Path variants
    kaggle_format_path = os.path.join(data_dir, 'captions.txt')
    token_format_path = os.path.join(data_dir, 'visium_token.txt')
    
    if os.path.exists(kaggle_format_path):
        print(f"Loading captions from Kaggle format: {kaggle_format_path}")
        with open(kaggle_format_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Skip header: image,caption
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    image_id = parts[0]
                    caption = ",".join(parts[1:]) # Rejoin if caption contained commas
                    cleaned = clean_text(caption)
                    if image_id not in captions_dict:
                        captions_dict[image_id] = []
                    captions_dict[image_id].append(cleaned)
                    
    elif os.path.exists(token_format_path):
        print(f"Loading captions from standard token format: {token_format_path}")
        with open(token_format_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_token = parts[0]
                    caption = parts[1]
                    # image_token is like "1000268201_693b08cb0e.jpg#0" -> extract image name
                    image_id = image_token.split('#')[0]
                    cleaned = clean_text(caption)
                    if image_id not in captions_dict:
                        captions_dict[image_id] = []
                    captions_dict[image_id].append(cleaned)
    else:
        raise FileNotFoundError(f"No caption files found in {data_dir}. Place custom captions first.")
        
    print(f"Loaded captions for {len(captions_dict)} images.")
    return captions_dict

def prepare_splits(captions_dict, data_dir):
    """
    Splits the dataset into Train, Validation, and Test sets.
    Uses custom train/val/test split files if present,
    otherwise falls back to an 80/10/10 random split.
    """
    train_ids_path = os.path.join(data_dir, 'train_images.txt')
    val_ids_path = os.path.join(data_dir, 'val_images.txt')
    test_ids_path = os.path.join(data_dir, 'test_images.txt')

    
    if os.path.exists(train_ids_path) and os.path.exists(val_ids_path) and os.path.exists(test_ids_path):
        print("Using custom dataset split text files.")
        with open(train_ids_path, 'r') as f:
            train_image_ids = [line.strip() for line in f if line.strip()]
        with open(val_ids_path, 'r') as f:
            val_image_ids = [line.strip() for line in f if line.strip()]
        with open(test_ids_path, 'r') as f:
            test_image_ids = [line.strip() for line in f if line.strip()]
    else:
        print("Custom dataset split files not found. Creating random 80/10/10 split.")
        all_ids = list(captions_dict.keys())
        train_image_ids, temp_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
        val_image_ids, test_image_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
        
    # Standardize to include key matches (some lists might omit '.jpg' extension)
    def ensure_jpg(id_list):
        return [i if i.lower().endswith('.jpg') else f"{i}.jpg" for i in id_list]
        
    train_image_ids = ensure_jpg(train_image_ids)
    val_image_ids = ensure_jpg(val_image_ids)
    test_image_ids = ensure_jpg(test_image_ids)
    
    # Filter dictionary based on split keys
    train_captions = {k: captions_dict[k] for k in train_image_ids if k in captions_dict}
    val_captions = {k: captions_dict[k] for k in val_image_ids if k in captions_dict}
    test_captions = {k: captions_dict[k] for k in test_image_ids if k in captions_dict}
    
    print(f"Dataset split size: Train={len(train_captions)}, Val={len(val_captions)}, Test={len(test_captions)}")
    return train_captions, val_captions, test_captions

def add_start_end_tokens(captions_dict):
    """
    Appends 'startseq' and 'endseq' tokens to captions.
    """
    formatted_dict = {}
    for image_id, captions in captions_dict.items():
        formatted_dict[image_id] = [f"startseq {c} endseq" for c in captions]
    return formatted_dict

def get_tokenizer(train_captions_dict, save_path=None):
    """
    Fits a Tokenizer on the training captions.
    """
    all_captions = []
    for captions in train_captions_dict.values():
        all_captions.extend(captions)
        
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer saved to {save_path}")
        
    return tokenizer

def load_tokenizer(load_path):
    with open(load_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded from {load_path}")
    return tokenizer

def build_numpy_arrays(captions_dict, image_id_to_idx, tokenizer, max_caption_length):
    """
    Pre-compute all (feature_idx, padded_seq, target_word) as tiny NumPy arrays.
    feature_idx is an integer index into the mmap list — no feature data duplicated.
    """
    feat_idx_list, seq_list, y_list = [], [], []

    for img_id, captions in captions_dict.items():
        if img_id not in image_id_to_idx:
            continue
        feat_idx = image_id_to_idx[img_id]
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq   = seq[:i]
                out_word = seq[i]
                pad_len  = max_caption_length - len(in_seq)
                in_seq   = [0] * pad_len + in_seq
                feat_idx_list.append(feat_idx)
                seq_list.append(in_seq)
                y_list.append(out_word)

    print(f"  {len(y_list):,} training pairs built.")
    return (
        np.array(feat_idx_list, dtype=np.int32),   # (N,)
        np.array(seq_list,      dtype=np.int32),   # (N, max_len)
        np.array(y_list,        dtype=np.int32),   # (N,)
    )

def make_dataset(captions_dict, features_array, image_id_to_idx, tokenizer, max_caption_length, batch_size):
    """
    Builds a tf.data.Dataset that is fully DirectML-compatible.
    Stores tiny (feature_idx, padded_seq, target_word) integer tuples and
    maps over them to fetch from the in-RAM features_array — no 202 GB allocation.
    """
    import tensorflow as tf

    # Pre-compute small index/sequence arrays (no feature duplication)
    feat_idx_list, seq_list, y_list = [], [], []

    for img_id, captions in captions_dict.items():
        if img_id not in image_id_to_idx:
            continue
        feat_idx = image_id_to_idx[img_id]

        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq  = seq[:i]
                out_word = seq[i]
                pad_len  = max_caption_length - len(in_seq)
                in_seq   = [0] * pad_len + in_seq
                feat_idx_list.append(feat_idx)
                seq_list.append(in_seq)
                y_list.append(out_word)

    print(f"  Built index dataset with {len(y_list):,} samples.")

    feat_idx_np = np.array(feat_idx_list, dtype=np.int32)   # shape (N,)
    seq_np      = np.array(seq_list,      dtype=np.int32)   # shape (N, max_len)
    y_np        = np.array(y_list,        dtype=np.int32)   # shape (N,)

    # Pin features to CPU RAM — only small batches get transferred to GPU
    with tf.device('/cpu:0'):
        features_tf = tf.constant(features_array, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((feat_idx_np, seq_np, y_np))
    dataset = dataset.shuffle(buffer_size=50000, reshuffle_each_iteration=True)

    def lookup_feature(feat_idx, seq, y):
        feature = features_tf[feat_idx]   # instant RAM lookup, no disk I/O
        return (feature, seq), y

    dataset = dataset.map(lookup_feature, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
