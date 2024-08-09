from flask import Flask, request, jsonify, render_template, make_response,url_for
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import base64
from PIL import Image
from io import BytesIO
import os
import googletrans
from googletrans import Translator
import gtts
from gtts import gTTS
import time
from IPython.display import Audio
app = Flask(__name__)
# Configuration




CORS(app)

language_codes = googletrans.LANGUAGES
languages = [{"code": code, "name": name} for code, name in language_codes.items()]
def translate_text(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text
def load_models():
    loaded_encoder = tf.keras.models.load_model(encoder_model_path)
    loaded_decoder = tf.keras.models.load_model(decoder_model_path)

    # Manually compile the models if required
    optimizer = tf.keras.optimizers.Adam()
    loaded_encoder.compile(optimizer=optimizer)
    loaded_decoder.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')  # Replace with your actual loss function

    return loaded_encoder, loaded_decoder



def convert_image_to_base64(image):
    print(".............")
    pil_image = Image.open(image).convert('RGB')
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_data



def load_image(image_path):    
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)    
    return img, image_path

# Image feature extraction model
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input 
hidden_layer = image_model.layers[-1].output 
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
# Load the pre-trained encoder and decoder models
encoder_model_path = './FinalModels/encoder_model'
decoder_model_path = './FinalModels/decoder_model'
encoder, decoder = load_models()
max_length = 20  # Adjust this as per your model
attention_features_shape = 64  # Adjust this as per your model
tokenizer_path = './FinalModels/tokenizer.pkl'
with open(tokenizer_path, 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)
def evaluate_V1(image, loaded_encoder, loaded_decoder, max_length, tokenizer, attention_features_shape):
    attention_plot = np.zeros((max_length, attention_features_shape))

    # Manually initialize the hidden state for the Decoder
    hidden = tf.zeros((1, loaded_decoder.layers[2].units))  # Adjust units based on your GRU layer

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = loaded_encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = loaded_decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot, predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot, predictions



      
@app.route("/", methods=["GET", "POST"])
def translate():
    if request.method == "POST":
        image = request.files['image']        
        target_language = request.form.get("target_language")        
        filename = os.path.join('./uploads', image.filename)
        image.save(filename)
        result, attention_plot, predictions = evaluate_V1(filename, encoder, decoder, max_length, tokenizer, attention_features_shape)
        result.remove('<end>')        
        caption = ' '.join(result)        
        translated_text = translate_text(caption, target_language)        
        timestamp = int(time.time())
        filename = f"static/op_{timestamp}.mp3"        
        tts = gTTS(translated_text, lang=target_language)
        tts.save(filename)
        server_address = "http://127.0.0.1:5000" #http://3.95.234.114/
        # audio_url = f"{server_address}/{filename}"
        audio_url = url_for('static', filename=f'op_{timestamp}.mp3')
        try:
            image_base64 = convert_image_to_base64(image)
            
        except Exception as e:
            return render_template('index.html', generation_message="Error processing image")
               
        return render_template("index.html", languages=languages, image=image_base64, caption=caption, translated_text=translated_text, audio_filename=audio_url)
    return render_template("index.html", languages=languages)


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    target_language = request.form.get('target_language', 'hi')
    print(target_language)
    filename = os.path.join('./uploads', file.filename)
    file.save(filename)    
    result, attention_plot, predictions = evaluate_V1(filename, encoder, decoder, max_length, tokenizer, attention_features_shape)
    if not result:
        return jsonify({
            'error': 'Unable to convert caption. Result is empty.'
        }), 400
    result.remove('<end>')
    caption = ' '.join(result)        
    translated_text = translate_text(caption, target_language)        
    timestamp = int(time.time())
    audio_filename = f"static/{target_language}_{timestamp}.mp3"  
    tts = gTTS(translated_text, lang=target_language)
    tts.save(audio_filename)
    # with open(audio_filename, 'rb') as audio_file:
    #     audio_bytes = audio_file.read()
    
    # audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    return jsonify({
        'caption': caption,
        'translated_text': translated_text,
        'audio_file': audio_filename
    })
    # return jsonify({'caption': ' '.join(result)})
    # return jsonify({
    #     'caption': caption,
    #     'translated_text': translated_text,
    #     'audio_file': audio_filename
    # })

   


if __name__ == "__main__":
    app.run(debug=True)

