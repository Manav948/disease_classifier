import sys, os, re, random
import json
import sqlite3
from collections import Counter
import numpy as np
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session, redirect, url_for
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import check_password_hash, generate_password_hash
import joblib

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-to-a-random-secret')

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, 'app.db')

# Load model and metadata
BASE = os.path.dirname(os.path.abspath(__file__))
clf = joblib.load(os.path.join(BASE, 'models', 'classifier.pkl'))
with open(os.path.join(BASE, 'models', 'symptoms.json')) as f:
    ALL_SYMPTOMS = json.load(f)
with open(os.path.join(BASE, 'models', 'disease_info.json')) as f:
    DISEASE_INFO = json.load(f)

SEVERITY_ORDER = {'Emergency': 5, 'High': 4, 'Chronic': 3, 'Moderate': 2, 'Mild': 1}

LANGUAGES = {
    'en': 'English',
    'hi': 'हिन्दी',
    'gu': 'ગુજરાતી'
}

TRANSLATIONS = {
    'en': {
        'site_title': 'MediScan - AI Disease Classifier',
        'diagnostic_console': 'Diagnostic Console',
        'health_chat': 'Health Chat',
        'history': 'History',
        'logout': 'Logout',
        'active_clinical_presentation': 'Active Clinical Presentation',
        'no_symptoms_selected': 'No symptoms selected. Choose indicators below to populate clinical profile.',
        'symptom_selector': 'Symptom Selector',
        'filter_clinical_indicators': 'Filter clinical indicators...',
        'respiratory_chest': 'Respiratory & Chest',
        'loading_symptoms': 'Loading symptoms...',
        'select_minimum_symptoms': 'Select a minimum of',
        'symptoms_lowercase': 'symptoms',
        'to_enable_precise_ml': 'to enable precise ML engine classification.',
        'general_presentation': 'General Presentation',
        'run_diagnostic_analysis': 'Run Diagnostic Analysis',
        'ai_classification_outcome': 'AI Classification Outcome',
        'analyzing_complex_symptom_clusters': 'Analyzing complex symptom clusters...',
        'diagnostic_output_pending': 'Diagnostic Output Pending',
        'output_pending_desc': 'Select symptoms on the left and trigger prediction to initialize engine diagnostics.',
        'download_pdf_report': 'Download PDF Report',
        'clinical_reference_guidelines': 'Clinical Reference Guidelines',
        'no_disease_selected': 'No Disease Selected',
        'click_predicted_disease': 'Click on any predicted disease above to view specialized medical guidelines here.',
        'sign_in_to_mediscane': 'Sign in to MediScan',
        'access_prediction_history': 'Access your prediction history and personalized health tracking.',
        'dont_have_account': 'Don\'t have an account? Register here',
        'back_to_home': 'Back to Home',
        'create_account': 'Create your MediScan account',
        'sign_up_prompt': 'Sign up to save your prediction history and access the dashboard.',
        'health_assistant_label': 'MediScan Health Assistant',
        'hero_title': 'AI-powered disease prediction. Analyze symptoms intelligently.',
        'hero_subtitle': 'Leveraging advanced Random Forest architectures to provide clinical-grade symptom analysis and probability-based diagnostic support in milliseconds.',
        'view_disease_database': 'View Disease Database',
        'nearby_hospitals': 'Nearby Hospitals & Clinics',
        'find_hospitals_description': 'Locate nearby emergency care facilities and clinics using your device location.',
        'find_nearby': 'Find Nearby',
        'maps_api_key_missing': 'Google Maps API key is not configured. Please set GOOGLE_MAPS_API_KEY in the environment.',
        'allow_location_access': 'Allow location access to display nearby hospitals.',
        'searching_nearby_hospitals': 'Searching nearby hospitals...',
        'no_nearby_hospitals_found': 'No nearby hospitals were found. Try moving to a different location or refresh the page.',
        'choose_language': 'Choose language',
        'emergency_alert_title': 'High-Risk Alert',
        'emergency_alert_description': 'Your selected symptoms indicate a potentially dangerous condition. Seek medical help immediately.',
        'already_have_account': 'Already have an account? Log in',
        'how_it_works': 'How it works',
        'chat_welcome': 'Hello! I’m your healthcare chatbot. Tell me what you’re feeling, and I’ll give you symptom guidance and doctor advice.',
        'prediction_history': 'Prediction History',
        'saved_medical_records': 'Saved Medical Records',
        'review_past_predictions': 'Review your past disease predictions, symptoms, specialist suggestions, and timestamped reports.',
        'no_saved_predictions': 'No saved predictions yet. Run the diagnostic console to store your first record.',
        'prediction_record': 'Prediction Record',
        'top_predictions': 'Top predictions',
        'symptoms_title': 'Symptoms',
        'username': 'Username',
        'password': 'Password',
        'confirm_password': 'Confirm Password',
        'sign_in': 'Sign In',
        'register': 'Register',
        'hello_health_assistant': 'AI Chatbot for Symptom Support',
        'ask_health_question': 'Describe your symptoms or ask a health question...',
        'talk_to_assistant': 'Talk to your health assistant',
        'chat_prompt': 'Type your symptoms, ask for precautions, or find out whether you should see a doctor.',
        'send': 'Send',
        'suggested_prompts': 'Suggested prompts'
    },
    'hi': {
        'site_title': 'MediScan - एआई रोग पहचान',
        'diagnostic_console': 'निदान कंसोल',
        'health_chat': 'स्वास्थ्य चैट',
        'history': 'इतिहास',
        'logout': 'लॉग आउट',
        'active_clinical_presentation': 'सक्रिय नैदानिक प्रस्तुति',
        'no_symptoms_selected': 'कोई लक्षण चयनित नहीं। कृपया नीचे संकेतों का चयन करें।',
        'symptom_selector': 'लक्षण चयनकर्ता',
        'filter_clinical_indicators': 'नैदानिक संकेत फ़िल्टर करें...',
        'respiratory_chest': 'श्वसन और छाती',
        'loading_symptoms': 'लक्षण लोड हो रहे हैं...',
        'select_minimum_symptoms': 'न्यूनतम चुनें',
        'symptoms_lowercase': 'लक्षण',
        'to_enable_precise_ml': 'सटीक एमएल इंजन वर्गीकरण को सक्षम करने के लिए।',
        'general_presentation': 'सामान्य प्रस्तुति',
        'run_diagnostic_analysis': 'नैदानिक विश्लेषण चलाएँ',
        'ai_classification_outcome': 'एआई वर्गीकरण परिणाम',
        'analyzing_complex_symptom_clusters': 'जटिल लक्षण समूहों का विश्लेषण किया जा रहा है...',
        'diagnostic_output_pending': 'नैदानिक आउटपुट लंबित',
        'output_pending_desc': 'नैदानिक विश्लेषण शुरू करने के लिए बाएँ लक्षणों का चयन करें।',
        'download_pdf_report': 'पीडीएफ रिपोर्ट डाउनलोड करें',
        'clinical_reference_guidelines': 'नैदानिक संदर्भ दिशानिर्देश',
        'no_disease_selected': 'कोई रोग चयनित नहीं',
        'click_predicted_disease': 'विशेष दवा दिशानिर्देश देखने के लिए किसी रोग पर क्लिक करें।',
        'sign_in_to_mediscane': 'MediScan में साइन इन करें',
        'access_prediction_history': 'अपना भविष्यवाणी इतिहास और स्वास्थ्य ट्रैकिंग देखें।',
        'dont_have_account': 'खाता नहीं है? यहाँ पंजीकरण करें',
        'back_to_home': 'मुख पृष्ठ पर जाएं',
        'create_account': 'अपना MediScan खाता बनाएं',
        'sign_up_prompt': 'अपना इतिहास सहेजने और डैशबोर्ड तक पहुँचने के लिए साइन अप करें।',
        'health_assistant_label': 'MediScan स्वास्थ्य सहायक',
        'hero_title': 'एआई-समर्थित रोग भविष्यवाणी। लक्षणों का बुद्धिमानी से विश्लेषण करें।',
        'hero_subtitle': 'उन्नत रैंडम फ़ॉरेस्ट आर्किटेक्चर का उपयोग करके क्लिनिकल-ग्रेड लक्षण विश्लेषण और संभावना-आधारित नैदानिक समर्थन मिली सेकंड में प्रदान करें।',
        'view_disease_database': 'रोग डेटाबेस देखें',
        'nearby_hospitals': 'निकट के अस्पताल और क्लिनिक',
        'find_hospitals_description': 'अपने डिवाइस स्थान का उपयोग करके पास के आपातकालीन देखभाल सुविधाओं और क्लिनिकों को खोजें।',
        'find_nearby': 'निकट खोजें',
        'maps_api_key_missing': 'Google मैप्स API कुंजी कॉन्फ़िगर नहीं है। कृपया वातावरण में GOOGLE_MAPS_API_KEY सेट करें।',
        'allow_location_access': 'पास के अस्पताल दिखाने के लिए स्थान एक्सेस की अनुमति दें।',
        'searching_nearby_hospitals': 'पास के अस्पताल खोज रहे हैं...',
        'no_nearby_hospitals_found': 'कोई पास के अस्पताल नहीं मिले। विभिन्न स्थान पर जाने का प्रयास करें या पृष्ठ को रिफ्रेश करें।',
        'choose_language': 'भाषा चुनें',
        'emergency_alert_title': 'उच्च-जोखिम चेतावनी',
        'emergency_alert_description': 'आपके चयनित लक्षण संभावित रूप से खतरनाक स्थिति सूचित करते हैं। तुरंत चिकित्सा सहायता प्राप्त करें।',
        'already_have_account': 'पहले से खाता है? लॉग इन करें',
        'how_it_works': 'यह कैसे काम करता है',
        'chat_welcome': 'नमस्ते! मैं आपका स्वास्थ्य चैटबॉट हूँ। मुझे बताएं कि आप कैसा महसूस कर रहे हैं, और मैं आपको लक्षण मार्गदर्शन और डॉक्टर की सलाह दूंगा।',
        'prediction_history': 'भविष्यवाणी इतिहास',
        'saved_medical_records': 'सहेजे गए मेडिकल रिकॉर्ड',
        'review_past_predictions': 'अपने पिछले रोग भविष्यवाणियों, लक्षणों, विशेषज्ञ सुझावों और समयवाचक रिपोर्ट की समीक्षा करें।',
        'no_saved_predictions': 'अभी तक कोई सहेजी गई भविष्यवाणियाँ नहीं। अपना पहला रिकॉर्ड संग्रहीत करने के लिए डायग्नोस्टिक कंसोल चलाएँ।',
        'prediction_record': 'भविष्यवाणी रिकॉर्ड',
        'top_predictions': 'शीर्ष भविष्यवाणियाँ',
        'symptoms_title': 'लक्षण',
        'username': 'उपयोगकर्ता नाम',
        'password': 'पासवर्ड',
        'confirm_password': 'पासवर्ड की पुष्टि करें',
        'sign_in': 'साइन इन करें',
        'register': 'पंजीकरण करें',
        'hello_health_assistant': 'लक्षण समर्थन के लिए एआई चैटबॉट',
        'ask_health_question': 'अपने लक्षणों का वर्णन करें या स्वास्थ्य प्रश्न पूछें...',
        'talk_to_assistant': 'अपने स्वास्थ्य सहायक से बात करें',
        'chat_prompt': 'अपने लक्षण टाइप करें, सुरक्षा के उपाय पूछें, या पूछें कि क्या आपको डॉक्टर को देखना चाहिए।',
        'send': 'भेजें',
        'suggested_prompts': 'प्रस्तावित प्रश्न'
    },
    'gu': {
        'site_title': 'MediScan - એઆઇ રોગ ઓળખ',
        'diagnostic_console': 'નિદાન કન્સોલ',
        'health_chat': 'હેલ્થ ચેટ',
        'history': 'ઇતિહાસ',
        'logout': 'લૉગ આઉટ',
        'back_to_home': 'મુખ પૃષ્ઠ પર જાઓ',
        'active_clinical_presentation': 'સક્રિય નૈदानિક પ્રસ્તુતિ',
        'no_symptoms_selected': 'કોઈ લક્ષણ પસંદ કરાયેલ નથી. કૃપા કરીને નીચે સૂચકો પસંદ કરો.',
        'symptom_selector': 'લક્ષણ પસંદકર્તા',
        'filter_clinical_indicators': 'નૈદાનિક સૂચકો ફિલ્ટર કરો...',
        'respiratory_chest': 'શ્વાસ અને છાતી',
        'loading_symptoms': 'લક્ષણો લોડ થઈ રહ્યાં છે...',
        'select_minimum_symptoms': 'ન્યૂનતમ પસંદ કરો',
        'symptoms_lowercase': 'લક્ષણો',
        'to_enable_precise_ml': 'સચોટ એમએલ એન્જિન વર્ગીકરણને સક્ષમ કરવા માટે.',
        'general_presentation': 'સામાન્ય પ્રસ્તુતિ',
        'run_diagnostic_analysis': 'નૈદાનિક વિશ્લેષણ ચલાવો',
        'ai_classification_outcome': 'એઆઇ વર્ગીકરણ પરિણામ',
        'analyzing_complex_symptom_clusters': 'જટિલ લક્ષણ ક્લસ્ટર્સનું વિશ્લેષણ થઈ રહ્યું છે...',
        'diagnostic_output_pending': 'નૈદાનિક આઉટપુટ બાકી છે',
        'output_pending_desc': 'નૈદાનિક વિશ્લેષણ શરૂ કરવા માટે ડાબેથી લક્ષણો પસંદ કરો.',
        'download_pdf_report': 'પીડીએફ રિપોર્ટ ડાઉનલોડ કરો',
        'clinical_reference_guidelines': 'નૈદાનિક સંદર્ભ માર્ગદર્શન',
        'no_disease_selected': 'કોઈ રોગ પસંદ કરાયો નથી',
        'click_predicted_disease': 'વિશેષ માર્ગદર્શન જોવા માટે કોઈ ભવિષ્યવાણી રોગ પર ક્લિક કરો.',
        'sign_in_to_mediscane': 'MediScanમાં સાઇન ઇન કરો',
        'access_prediction_history': 'તમારા ભવિષ્યવાણી ઇતિહાસ અને આરોગ્ય ટ્રેકિંગ ઍક્સેસ કરો.',
        'dont_have_account': 'એકાઉન્ટ નથી? અહીં નોંધણી કરો',
        'create_account': 'તમારું MediScan એકાઉન્ટ બનાવો',
        'sign_up_prompt': 'તમારો ઇતિહાસ સાચવવા અને ડેશબોર્ડ પર જવા માટે સાઇન અપ કરો.',
        'health_assistant_label': 'MediScan હેલ્થ સહાયતાસભર',
        'hero_title': 'એઆઇ-સક્ષમ રોગ ભવિષ્યવાણી. લક્ષણોને બુદ્ધિમત્તાપૂર્વક વિશ્લેષણ કરો.',
        'hero_subtitle': 'ઉન્નત રેન્ડમ ફોરેસ્ટ આર્કિટેક્ચર્સને ઉપયોગ કરીને ક્લિનિકલ-ગ્રેડ લક્ષણ વિશ્લેષણ અને સંભાવના-આધારિત નિદાનાત્મક સપોર્ટ મિલિસેકંડ્સમાં પૂરો પાડે છે.',
        'view_disease_database': 'રોગ ડેટાબેસ જુઓ',
        'nearby_hospitals': 'નજીકના હોસ્પિટલ અને ક્લિનિક',
        'find_hospitals_description': 'તમારા ડિવાઈસ સ્થાનનો ઉપયોગ કરીને નજીકનાં તાત્કાલિક સારવાર કેન્દ્રો અને ક્લિનિક શોધો.',
        'find_nearby': 'નજીક શોધો',
        'maps_api_key_missing': 'Google મેપ્સ API કી રૂપરેખાંકિત નથી. કૃપા કરીને પર્યાવરણમાં GOOGLE_MAPS_API_KEY સેટ કરો.',
        'allow_location_access': 'નજીકના હોસ્પિટલ બતાવવા માટે સ્થાન ઍક્સેસની મંજૂરી આપો.',
        'searching_nearby_hospitals': 'નજીકના હોસ્પિટલ શોધી રહ્યા છીએ...',
        'no_nearby_hospitals_found': 'કોઈ નજીકના હોસ્પિટલ મળ્યા નથી. અલગ સ્થાન પર જવા અથવા પૃષ્ઠને રિફ્રેશ કરવાનો પ્રયાસ કરો.',
        'choose_language': 'ભાષા પસંદ કરો',
        'emergency_alert_title': 'ઉચ્ચ જોખમ સૂચના',
        'emergency_alert_description': 'તમારા પસંદ કરેલા લક્ષણો સંભવિત રીતે જોખમભરી સ્થિતિ દર્શાવે છે. તરત જ તબીબી મદદ મેળવો.',
        'already_have_account': 'મૂંડી ખાતું છે? લૉગ ઇન કરો',
        'how_it_works': 'આ કેવી રીતે કાર્ય કરે છે',
        'chat_welcome': 'હેલો! હું તમારો આરોગ્ય ચેટબોટ છું. મને જણાવો કે તમને કેવું લાગે છે, અને હું તમને લક્ષણ માર્ગદર્શન અને ડોક્ટરની સલાહ આપશે.',
        'prediction_history': 'ભવિષ્યવાણી ઇતિહાસ',
        'saved_medical_records': 'સાચવેલ મેડિકલ રેકોર્ડ',
        'review_past_predictions': 'તમારી ભૂતપૂર્વ રોગ ભવિષ્યવાણીઓ, લક્ષણો, નિષ્ણાત સૂચનો અને સમયમુકત રિપોર્ટ્સની સમીક્ષા કરો.',
        'no_saved_predictions': 'હજુ સુધી કોઈ સાચવેલી ભવિષ્યવાણીઓ નથી. તમારો પ્રથમ રેકોર્ડ સંગ્રહિત કરવા માટે ડાયગ્નોસ્ટિક કન્સોલ ચલાવો.',
        'prediction_record': 'ભવિષ્યવાણી રેકોર્ડ',
        'top_predictions': 'શ્રેષ્ઠ ભવિષ્યવાણીઓ',
        'symptoms_title': 'લક્ષણો',
        'username': 'વપરાશકર્તાનું નામ',
        'password': 'પાસવર્ડ',
        'confirm_password': 'પાસવર્ડની પુષ્ટિ કરો',
        'sign_in': 'સાઇન ઇન',
        'register': 'નોંધણી કરો',
        'hello_health_assistant': 'લક્ષણ સમર્થન માટે એઆઇ ચેટબોટ',
        'ask_health_question': 'તમારા લક્ષણો વર્ણવો અથવા આરોગ્ય પ્રશ્ન પુછો...',
        'talk_to_assistant': 'તમારા આરોગ્ય સહાયક સાથે વાત કરો',
        'chat_prompt': 'તમારા લક્ષણો ટાઇપ કરો, કુટુંબની સલાહ માંગો અથવા પૂછો કે શું તમને ડોકਟਰને મળવું જોઈએ.',
        'send': 'મુકલો',
        'suggested_prompts': 'સૂચવેલી સૂચનો'
    }
}


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_locale():
    lang = session.get('lang', 'en')
    if lang not in LANGUAGES:
        lang = 'en'
    return lang


@app.before_request
def ensure_language():
    lang = request.args.get('lang')
    if lang in LANGUAGES:
        session['lang'] = lang
    if 'lang' not in session:
        session['lang'] = 'en'


@app.route('/set_language/<lang_code>')
def set_language(lang_code):
    if lang_code in LANGUAGES:
        session['lang'] = lang_code
    return redirect(request.referrer or url_for('index'))


@app.context_processor
def inject_translations():
    lang = get_locale()
    trans = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    return {
        't': lambda key: trans.get(key, key),
        'lang': lang,
        'languages': LANGUAGES
    }


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    ''')
    # Add email column to existing DBs that don't have it yet
    try:
        cursor.execute('ALTER TABLE users ADD COLUMN email TEXT;')
    except Exception:
        pass
    conn.commit()
    conn.close()


@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        if not username or not password:
            return render_template('login.html', error='All fields are required.')

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?;', (username,)).fetchone()
        conn.close()
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))

        return render_template('login.html', error='Invalid username or password.')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm', '')

        if not username or not password or not confirm:
            return render_template('register.html', error='All fields are required.')
        if password != confirm:
            return render_template('register.html', error='Passwords do not match.')

        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?);', (
                username,
                email,
                generate_password_hash(password),
                datetime.utcnow().isoformat()
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return render_template('register.html', error='Username already exists.')
        conn.close()
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/app')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template(
        'app.html',
        username=session.get('username'),
        # Removed Google Maps integration: no API key passed
    )


@app.route('/api/symptoms')
def get_symptoms():
    return jsonify({'symptoms': ALL_SYMPTOMS})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected = data.get('symptoms', [])
    if not selected:
        return jsonify({'error': 'No symptoms selected'}), 400

    results = run_prediction(selected)
    if isinstance(results, tuple) and results[1] != 200:
        return jsonify(results[0]), results[1]

    emergency_info = detect_emergency_risk(selected, results[:3])

    return jsonify({
        'predictions': results[:3],
        'selected_count': len(selected),
        'disclaimer': 'This is an AI-based tool for informational purposes only. Always consult a qualified healthcare professional for medical advice.',
        'emergency_alert': emergency_info['emergency_alert'],
        'emergency_reason': emergency_info['emergency_reason'],
        'risk_level': emergency_info['risk_level']
    })


def run_prediction(selected):
    # Accept either space-separated symptom names (user text) or underscored symptom keys
    if not selected:
        return ({'error': 'No symptoms provided'}, 400)

    # Normalize selected symptom names to match ALL_SYMPTOMS entries
    normalized_selected = []
    for s in selected:
        if isinstance(s, str):
            key = s.replace(' ', '_') if ' ' in s else s
            # If provided as user-friendly (spaces), convert to underscored key if exists
            if key in ALL_SYMPTOMS:
                normalized_selected.append(key)
            else:
                # try replacing underscores/spaces both ways
                alt = key.replace('_', ' ')
                if alt in ALL_SYMPTOMS:
                    normalized_selected.append(alt)
                else:
                    # check for direct match ignoring underscores
                    for known in ALL_SYMPTOMS:
                        if known.replace('_', ' ') == s or known == s:
                            normalized_selected.append(known)
                            break

    if not normalized_selected:
        return ({'error': 'No valid symptoms matched'}, 400)

    # Build feature vector
    vec = [0] * len(ALL_SYMPTOMS)
    for s in normalized_selected:
        if s in ALL_SYMPTOMS:
            vec[ALL_SYMPTOMS.index(s)] = 1

    X = np.array([vec])
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_

    top_idx = np.argsort(proba)[::-1][:5]
    results = []
    for i in top_idx:
        disease = classes[i]
        prob = float(proba[i])
        if prob < 0.01:
            continue
        info = DISEASE_INFO.get(disease, {})
        matched = [s for s in normalized_selected if s in info.get('symptoms', [])]
        results.append({
            'disease': disease,
            'probability': round(prob * 100, 1),
            'description': info.get('description', ''),
            'severity': info.get('severity', 'Unknown'),
            'action': info.get('action', ''),
            'matched_symptoms': matched,
            'total_symptoms': len(info.get('symptoms', [])),
            'medicines': info.get('medicines', []),
            'precautions': info.get('precautions', []),
            'home_remedies': info.get('home_remedies', []),
            'specialist': info.get('specialist', 'General Physician')
        })

    if not results:
        return ({'error': 'Could not determine a diagnosis'}, 400)

    return results

@app.route('/api/diseases')
def get_diseases():
    diseases = []
    for name, info in DISEASE_INFO.items():
        diseases.append({
            'name': name,
            'severity': info.get('severity'),
            'symptom_count': len(info.get('symptoms', [])),
            'description': info.get('description', '')
        })
    diseases.sort(key=lambda x: SEVERITY_ORDER.get(x['severity'], 0), reverse=True)
    return jsonify({'diseases': diseases, 'total': len(diseases)})



def normalized_text(text):
    return re.sub(r'[^a-z0-9 ]+', ' ', text.lower()).strip()

EMERGENCY_TERMS = [
    'chest pain', 'shortness of breath', 'difficulty breathing', 'unconscious',
    'fainting', 'bleeding', 'severe pain', 'sudden weakness', 'rapid heartbeat',
    'collapsed', 'sharp pain', 'confusion', 'loss of consciousness', 'stroke',
    'seizure', 'panic attack', 'poisoning', 'severe allergic', 'allergic reaction'
]
DOCTOR_KEYWORDS = ['doctor', 'physician', 'urgent care', 'hospital', 'medical help', 'consult', 'specialist', 'emergency', 'see a doctor', 'appointment']
PRECAUTION_KEYWORDS = ['prevent', 'precaution', 'avoid', 'safety', 'reduce risk', 'stay safe', 'hygiene', 'sanitize', 'hand wash', 'mask', 'vaccinate', 'vaccine']
GREETING_KEYWORDS = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon']
THANK_YOU_KEYWORDS = ['thank', 'thanks', 'appreciate', 'ty']

EMERGENCY_SYMPTOMS = [
    'chest_pain', 'shortness_of_breath', 'difficulty_breathing', 'unconscious',
    'fainting', 'bleeding', 'severe_pain', 'sudden_weakness', 'rapid_heartbeat',
    'high_fever', 'severe_headache', 'rash', 'vomiting', 'confusion'
]


def detect_emergency_risk(selected_symptoms, predictions):
    emergency_symptoms = [s for s in selected_symptoms if s in EMERGENCY_SYMPTOMS]
    top_severity = predictions[0]['severity'] if predictions else 'Unknown'
    top_severity_rank = SEVERITY_ORDER.get(top_severity, 0)
    emergency_alert = bool(emergency_symptoms or top_severity_rank >= SEVERITY_ORDER.get('High', 4))

    reasons = []
    if emergency_symptoms:
        pretty_symptoms = ', '.join([s.replace('_', ' ') for s in emergency_symptoms])
        reasons.append(f'Detected potentially dangerous symptoms: {pretty_symptoms}.')
    if top_severity_rank >= SEVERITY_ORDER.get('High', 4):
        reasons.append(f'Top predicted condition carries a {top_severity} severity rating.')

    return {
        'emergency_alert': emergency_alert,
        'emergency_reason': ' '.join(reasons) if reasons else 'No high-risk symptoms detected.',
        'risk_level': top_severity
    }


def find_symptoms_in_text(text):
    found = []
    for symptom in ALL_SYMPTOMS:
        normalized = symptom.replace('_', ' ')
        if normalized in text:
            found.append(normalized)
    return found


def append_chat_history(sender, text, meta=None):
    hist = session.get('chat_history', [])
    entry = {'sender': sender, 'text': text}
    if meta is not None:
        entry['meta'] = meta
    hist.append(entry)
    session['chat_history'] = hist
    session.modified = True


@app.route('/api/chat/history')
def chat_history_api():
    hist = session.get('chat_history', [])
    collected = session.get('chat_symptoms', [])
    # present collected symptoms as readable strings
    return jsonify({
        'history': hist,
        'collected_symptoms': [s.replace('_', ' ') for s in collected]
    })

def find_disease_in_text(text):
    for disease in DISEASE_INFO:
        if disease.lower() in text:
            return disease
    return None

def build_disease_summary(disease):
    info = DISEASE_INFO.get(disease, {})
    return (
        f"{disease} is usually described as {info.get('description', 'a medical condition')}. "
        f"Severity is often classified as {info.get('severity', 'Unknown')}. "
        f"Recommended action: {info.get('action', 'Consult a qualified healthcare provider for an accurate diagnosis.')}."
    )

def generate_chat_reply(user_query):
    message = normalized_text(user_query)
    if not message:
        return {
            'reply': 'Please tell me your symptoms or ask a health question so I can help.',
            'doctor_recommendation': False,
            'risk_level': 'low'
        }

    if any(term in message for term in GREETING_KEYWORDS):
        return {
            'reply': 'Hello! I am your health assistant. Please describe your symptoms or ask about precautions and whether you should consult a doctor.',
            'doctor_recommendation': False,
            'risk_level': 'low'
        }

    if any(term in message for term in THANK_YOU_KEYWORDS):
        return {
            'reply': 'You’re welcome! If you have more symptoms or questions, I am here to help.',
            'doctor_recommendation': False,
            'risk_level': 'low'
        }

    if any(term in message for term in EMERGENCY_TERMS):
        return {
            'reply': 'Your description includes warning signs that should be taken seriously. Please seek immediate medical attention or contact emergency services right away.',
            'doctor_recommendation': True,
            'risk_level': 'high'
        }

    disease_name = find_disease_in_text(message)
    if disease_name:
        return {
            'reply': build_disease_summary(disease_name),
            'doctor_recommendation': True,
            'risk_level': DISEASE_INFO.get(disease_name, {}).get('severity', 'Unknown')
        }

    symptoms = find_symptoms_in_text(message)
    if symptoms:
        top_matches = []
        for disease, info in DISEASE_INFO.items():
            matched = [s for s in symptoms if s in [x.replace('_', ' ') for x in info.get('symptoms', [])]]
            if matched:
                top_matches.append((disease, len(matched), info.get('severity', 'Unknown')))
        top_matches.sort(key=lambda x: (x[1], SEVERITY_ORDER.get(x[2], 0)), reverse=True)

        if top_matches:
            suggestions = []
            for disease, match_count, severity in top_matches[:2]:
                suggestions.append(f"{disease} ({severity}, matched {match_count} symptom(s))")
            reply = (
                f"I detected these symptom signals: {', '.join(symptoms)}. "
                f"This may correspond to conditions such as {', '.join(suggestions)}. "
                "Rest, stay hydrated, and monitor your symptoms closely. "
                "If your symptoms worsen, if you develop fever, breathing difficulty, or severe pain, please consult a doctor."
            )
            doctor_recommendation = any(SEVERITY_ORDER.get(info.get('severity', ''), 0) >= 4 for disease, _, _ in top_matches[:2] for info in [DISEASE_INFO.get(disease, {})])
            return {
                'reply': reply,
                'doctor_recommendation': doctor_recommendation,
                'risk_level': top_matches[0][2] if top_matches else 'Moderate'
            }

    if any(term in message for term in PRECAUTION_KEYWORDS):
        tips = [
            'Keep your hands clean by washing regularly and using sanitizer.',
            'Stay hydrated, eat balanced meals, and get adequate rest.',
            'Avoid crowded places if you feel unwell, and wear a mask when needed.',
            'Seek medical advice if symptoms last more than a few days or worsen.'
        ]
        return {
            'reply': 'Here are a few general health precautions: ' + ' '.join(tips),
            'doctor_recommendation': False,
            'risk_level': 'low'
        }

    if any(term in message for term in DOCTOR_KEYWORDS):
        return {
            'reply': 'If you are unsure about your symptoms or they are persistent, it is best to consult a healthcare professional. A doctor can provide a personalized diagnosis and recommend treatment.',
            'doctor_recommendation': True,
            'risk_level': 'moderate'
        }

    general_tips = [
        'Describe how long the symptoms have been present and if they are getting better or worse.',
        'Pay attention to fever, breathing difficulty, chest pain, or severe headaches.',
        'Rest and keep hydrated, but seek medical care if you feel the condition is worsening.'
    ]
    return {
        'reply': 'I am here to help with symptom guidance and doctor recommendations. ' + random.choice(general_tips),
        'doctor_recommendation': False,
        'risk_level': 'low'
    }

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()
    if not query:
        return jsonify({'error': 'Please enter a health question or symptom description.'}), 400
    # record user's message in conversation memory
    append_chat_history('user', query)

    # Ensure session storage for chat-collected symptoms
    collected = session.get('chat_symptoms', [])

    # extract symptoms from user message
    message_norm = normalized_text(query)
    found = find_symptoms_in_text(message_norm)  # returns list with spaces
    found_keys = [s.replace(' ', '_') for s in found]

    for s in found_keys:
        if s not in collected:
            collected.append(s)

    session['chat_symptoms'] = collected

    # If user message contains emergency keywords, short-circuit to urgent reply
    if any(term in message_norm for term in EMERGENCY_TERMS):
        session.pop('chat_symptoms', None)
        assistant_text = 'Your description includes warning signs that should be taken seriously. Please seek immediate medical attention or contact emergency services right away.'
        append_chat_history('assistant', assistant_text, meta={'doctor_recommendation': True, 'risk_level': 'high'})
        return jsonify({
            'reply': assistant_text,
            'doctor_recommendation': True,
            'risk_level': 'high'
        })

    # If we have at least 2 collected symptoms, run prediction
    if len(collected) >= 2:
        results = run_prediction(collected)
        if isinstance(results, tuple) and results[1] != 200:
            return jsonify(results[0]), results[1]

        # build assistant reply summarizing top predictions
        top = results[:3]
        preds_text = '; '.join([f"{p['disease']} ({p['probability']}%)" for p in top])
        assistant_text = f"I detected these symptoms: {', '.join([s.replace('_', ' ') for s in collected])}. Top predictions: {preds_text}. {top[0].get('action', '')}"

        # record assistant reply and predictions in history
        append_chat_history('assistant', assistant_text, meta={'predictions': top, 'selected_symptoms': [s.replace('_', ' ') for s in collected]})

        # clear collected symptoms after prediction to reset context
        session.pop('chat_symptoms', None)

        return jsonify({
            'reply': assistant_text,
            'predictions': top,
            'selected_symptoms': [s.replace('_', ' ') for s in collected],
            'doctor_recommendation': any(p.get('severity') == 'High' for p in top),
            'risk_level': top[0].get('severity', 'Unknown')
        })

    # If not enough symptoms, ask a follow-up question to collect more details
    follow_up_questions = [
        'Do you have a fever or chills?',
        'Are you experiencing cough or sore throat?',
        'Any shortness of breath or chest discomfort?',
        'Do you have nausea, vomiting, or diarrhea?',
        'How long have you had these symptoms?'
    ]
    ask = follow_up_questions[0]

    # use the generic generator for non-symptom small talk
    gen = generate_chat_reply(query)
    reply = gen.get('reply', '')
    combined = reply + ' ' + ask

    append_chat_history('assistant', combined, meta={'follow_up': True, 'ask': ask, 'collected_symptoms': [s.replace('_', ' ') for s in collected]})

    return jsonify({
        'reply': combined,
        'follow_up': True,
        'ask': ask,
        'collected_symptoms': [s.replace('_', ' ') for s in collected]
    })


def draw_wrapped_text(canvas_obj, text, x, y, max_width, line_height, font_name='Helvetica', font_size=10):
    words = str(text).split()
    line = ''
    for word in words:
        test_line = f"{line} {word}".strip()
        if stringWidth(test_line, font_name, font_size) <= max_width:
            line = test_line
        else:
            canvas_obj.drawString(x, y, line)
            y -= line_height
            line = word
        if y < 60:
            canvas_obj.showPage()
            y = letter[1] - 50
            canvas_obj.setFont(font_name, font_size)
    if line:
        canvas_obj.drawString(x, y, line)
        y -= line_height
    return y


@app.route('/api/report', methods=['POST'])
def generate_report():
    data = request.get_json(silent=True) or {}
    predictions = data.get('predictions', [])
    symptoms = data.get('symptoms', [])
    selected_count = data.get('selected_count', 0)

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin

    pdf.setFont('Helvetica-Bold', 18)
    pdf.drawString(margin, y, 'MediScan Health Report')
    y -= 26
    pdf.setFont('Helvetica', 10)
    pdf.drawString(margin, y, f'Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    y -= 18
    pdf.drawString(margin, y, f'Symptom count: {selected_count}')
    y -= 14
    pdf.drawString(margin, y, 'Symptoms: ' + (', '.join(symptoms) if symptoms else 'N/A'))
    y -= 24

    if not predictions:
        pdf.drawString(margin, y, 'No prediction data available. Please run the diagnostic analysis first.')
    else:
        pdf.setFont('Helvetica-Bold', 14)
        pdf.drawString(margin, y, 'Prediction Summary')
        y -= 20

        for index, pred in enumerate(predictions, start=1):
            if y < margin + 120:
                pdf.showPage()
                y = height - margin
                pdf.setFont('Helvetica-Bold', 14)
                pdf.drawString(margin, y, 'Prediction Summary (continued)')
                y -= 24

            pdf.setFont('Helvetica-Bold', 12)
            pdf.drawString(margin, y, f'{index}. {pred.get("disease", "Unknown")} ({pred.get("probability", 0)}%)')
            y -= 16
            pdf.setFont('Helvetica', 10)
            pdf.drawString(margin + 12, y, f"Severity: {pred.get('severity', 'Unknown')}")
            pdf.drawString(margin + 240, y, f"Specialist: {pred.get('specialist', 'General Physician')}")
            y -= 14
            y = draw_wrapped_text(pdf, f"Description: {pred.get('description', '')}", margin + 12, y, width - margin * 2 - 12, 12)
            y = draw_wrapped_text(pdf, f"Recommended action: {pred.get('action', '')}", margin + 12, y, width - margin * 2 - 12, 12)

            medicines = pred.get('medicines', [])
            precautions = pred.get('precautions', [])
            home_remedies = pred.get('home_remedies', [])

            if medicines:
                y = draw_wrapped_text(pdf, 'Medicines: ' + ', '.join(medicines), margin + 12, y, width - margin * 2 - 12, 12)
            if precautions:
                y = draw_wrapped_text(pdf, 'Precautions: ' + ', '.join(precautions), margin + 12, y, width - margin * 2 - 12, 12)
            if home_remedies:
                y = draw_wrapped_text(pdf, 'Home remedies: ' + ', '.join(home_remedies), margin + 12, y, width - margin * 2 - 12, 12)

            y -= 12

    pdf.save()
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name='MediScan_Report.pdf',
        mimetype='application/pdf'
    )


if __name__ == '__main__':
    init_db()  # Ensure DB tables exist on every startup
    print("=" * 60)
    print("  Disease Classifier — Running at http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, port=5000)
