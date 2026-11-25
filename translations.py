"""
Translations module for SugarSense Diabetes Risk Prediction Web App.
Supports English ('en') and Bahasa Indonesia ('id').

The Indonesian text is written to sound natural to Indonesian readers
(everyday health vocabulary such as "gula darah" and "darah tinggi"),
not as a word-for-word translation of the English text.
"""

TRANSLATIONS = {
    'en': {
        # App & Navigation
        'app_title': 'SugarSense',
        'lang_label': '🌐 Language / Bahasa',
        'nav_dashboard': 'Dashboard',
        'nav_prediction': 'Prediction',
        'nav_report': 'Report',
        'greet_morning': 'Good Morning',
        'greet_afternoon': 'Good Afternoon',
        'greet_evening': 'Good Evening',
        'greet_name': 'there!',
        'btn_check_new': '+  Check new',
        'chip_features': 'Model features',
        'chip_threshold': 'Decision threshold',
        'chip_accuracy': 'Model accuracy',
        'chip_dataset': 'Dataset',
        'chip_records': 'records',
        'card_glucose_trend': 'Glucose Trend by Age',
        'card_prevalence': 'Diabetes Prevalence',
        'card_recent': 'Recent Checks',
        'card_health_info': 'Health Information',
        'card_bmi_rate': 'Diabetes Rate by BMI Category',
        'card_ranges': 'Healthy Ranges',
        'pill_dataset': 'Dataset',
        'pill_today': "Today's info",
        'pill_last_input': 'Last input',
        'pill_model': 'Model',
        'pill_guide': 'Guide',
        'pill_tips': 'Tips',
        'pill_fields': '14 fields',
        'pill_model_output': 'Model output',
        'pill_personalized': 'Personalized',
        'pill_how_to_read': 'How to read',
        'pill_result': 'Result',
        'of_people': 'of {n:,} people',
        'no_checks_title': 'No checks yet',
        'no_checks_sub': 'Run your first prediction to see it here.',
        'latest_check': 'Your latest check',
        'below_avg': 'Below average',
        'above_avg': 'Above average',
        'avg_label': 'avg',
        'step_1': '① Fill data',
        'step_2': '② Generate prediction',
        'step_3': '③ Read your report',
        'tips_title': 'Before you start',
        'tips_desc': 'Use your most recent measurements. Fasting glucose and resting blood pressure give the most reliable prediction.',
        'card_your_metrics': 'Your Health Metrics',
        'card_pred_prob': 'Prediction Probability',
        'report_subtitle': 'Here is the summary of your diabetes risk prediction.',
        'chip_risk': 'Diabetes risk',
        'neg_tip1_title': 'Stay active',
        'neg_tip1_sub': 'Aim for 150 minutes of exercise weekly.',
        'neg_tip2_title': 'Regular check-ups',
        'neg_tip2_sub': 'Keep monitoring your health yearly.',
        'pos_tip1_title': 'Consult a doctor',
        'pos_tip1_sub': 'We highly recommend a medical check-up.',
        'pos_tip2_title': 'Review your lifestyle',
        'pos_tip2_sub': 'Healthy diet, exercise, and regular monitoring.',
        'kv_algorithm': 'Algorithm',
        'kv_features': 'Input features',
        'kv_features_val': '14 indicators',
        'kv_threshold': 'Threshold',
        'kv_accuracy': 'Test accuracy',
        'kv_training': 'Training data',
        'kv_output': 'Output',
        'range_glucose': 'Glucose',
        'range_bmi': 'Body Mass Index',
        'range_bp': 'Blood Pressure',
        'range_pulse': 'Pulse Rate',
        
        # Dashboard / Home
        'hero_title': 'Good Day!',
        'hero_subtitle': 'Welcome to your early diabetes risk prediction dashboard.',
        'what_is_title': 'What is SugarSense?',
        'what_is_desc': (
            'SugarSense is a simple AI-powered web application designed to help users '
            'estimate their diabetes risk based on basic health indicators. By analyzing '
            'information such as glucose level, BMI, and blood pressure, the system provides '
            'an early prediction to encourage better health awareness and lifestyle habits.'
        ),
        'what_do_title': 'What SugarSense do?',
        'what_do_desc': (
            'We use machine learning technology to analyze your health data and '
            'generate a quick diabetes risk prediction in seconds.'
        ),
        'why_title': 'Why SugarSense?',
        'why_desc': (
            'Early awareness can make a big difference. SugarSense aims to make health '
            'screening more accessible, simple, and easy to understand for everyone.'
        ),
        'disclaimer_title': '⚠️ Educational Purposes Only',
        'disclaimer_desc': (
            'SugarSense is intended for educational purposes only and should not replace professional '
            'medical advice or diagnosis. Always consult a healthcare professional.'
        ),
        'btn_start': '+ Start New Prediction',

        # Input Page
        'form_title': 'Health Information',
        'form_subtitle': 'Please fill in your current health metrics to get an accurate prediction.',
        'label_age': 'Age (years)',
        'help_age': 'Your current age in years',
        'label_gender': 'Gender',
        'opt_male': 'Male',
        'opt_female': 'Female',
        'label_height': 'Height (cm)',
        'help_height': 'Enter your height in centimeters, e.g. 160',
        'label_glucose': 'Glucose Level (mmol/L)',
        'help_glucose': 'Normal range: 4.0-7.0 mmol/L',
        'label_hypertensive': 'Hypertensive',
        'label_family_diabetes': 'Family History of Diabetes',
        'help_family_diabetes': 'Do any close relatives have diabetes?',
        'label_family_hypertension': 'Family History of Hypertension',
        'help_family_hypertension': 'Do any close relatives have hypertension?',
        'label_cardiovascular': 'Cardiovascular Disease',
        'help_cardiovascular': 'Have you been diagnosed with heart or cardiovascular disease?',
        'label_stroke': 'History of Stroke',
        'help_stroke': 'Have you ever had a stroke?',
        'opt_no': 'No',
        'opt_yes': 'Yes',
        'label_systolic': 'Systolic Blood Pressure (mmHg)',
        'help_systolic': 'Normal: <120 mmHg',
        'label_diastolic': 'Diastolic Blood Pressure (mmHg)',
        'help_diastolic': 'Normal: <80 mmHg',
        'label_bmi': 'Body Mass Index (BMI)',
        'help_bmi': 'Normal range: 18.5-24.9',
        'label_weight': 'Weight (kg)',
        'label_pulse': 'Pulse Rate (bpm)',
        'help_pulse': 'Normal range: 60-100 bpm',
        'btn_predict': 'Generate Prediction',
        'err_zero': '⚠️ Please enter valid values for all fields. Values cannot be 0.',
        'err_model': '❌ Model not found. Please ensure model.pkl exists in the directory.',

        # Result Pages
        'report_heading': 'Prediction Report',
        'neg_title': 'No Diabetes Detected',
        'neg_subtitle': 'Your health indicators currently show a low likelihood of diabetes.',
        'pos_title': 'Diabetes Detected',
        'pos_subtitle': 'The model identified indicators associated with diabetes.',
        'chart_safe': 'Safe',
        'chart_risk': 'Risk',
        'chart_non_diab': 'Non-Diabetes',
        'chart_diab': 'Diabetes',
        'analysis_title': 'Health Status Analysis',
        'neg_analysis_desc': (
            'Based on your health data, the system detected a low possibility of diabetes. '
            'The chart on the left illustrates your current prediction confidence generated by our AI model. '
            'Keep maintaining your healthy habits and continue taking care of your body.'
        ),
        'neg_highlight': '💡 Keep up your healthy lifestyle and regular health check-ups.',
        'pos_analysis_desc': (
            'Based on your health data, you have a <b>{prob:.0f}% possibility</b> of diabetes. '
            'The chart on the left illustrates your current prediction confidence generated by our AI model.<br><br>'
            'Don\'t panic — this is an early algorithmic prediction to help you stay more aware of your health.'
        ),
        'pos_highlight': '💡 We highly recommend consulting a healthcare professional.',
        'btn_new_pred': 'New Prediction',

        # Result explanation (what the numbers mean - short & plain)
        'explain_title': '📖 How to Read Your Result',
        'explain_intro': 'A quick guide to what each number means.',
        'explain_score_title': 'Your Risk Score',
        'explain_score_desc': 'Your estimated chance of diabetes from the AI model. Higher means higher risk (not a medical diagnosis).',
        'explain_safe_note': '"Safe" on the chart is simply 100% minus this score.',
        'explain_threshold_title': 'Decision Threshold',
        'explain_threshold_desc': 'The cut-off line. A score at or above this line is flagged as Diabetes.',
        'explain_threshold_above': 'Your score ({score:.0f}%) is above the line → {label}.',
        'explain_threshold_below': 'Your score ({score:.0f}%) is below the line → {label}.',
        'explain_accuracy_title': 'Model Accuracy',
        'explain_accuracy_desc': 'In tests on {n:,} people, the model got about {correct:.0f} out of 100 right.',
        'explain_note_title': 'Quick note on accuracy',
        'explain_note_desc': 'Only about {share:.0f} in 100 people in the dataset had diabetes, so two extra numbers also matter:',
        'explain_recall': '<b>Recall {recall:.0f}%</b>: catches about {recall:.0f} out of 100 real diabetes cases.',
        'explain_precision': '<b>Precision {precision:.0f}%</b>: about {precision:.0f} in 100 positive alerts truly had diabetes — treat a positive result as a reminder to see a doctor.',

        # Interactive Insights
        'insights_title': '📌 Interactive Health Insights',
        'tab_deep_dive': '🔍 Deep Dive Insights',
        'tab_gauges': '📊 Visual Gauges',
        'tab_simulator': '🎛️ What-If Simulator',

        # Deep Dive Cards
        'card_bmi_title': '⚖️ Weight Management (BMI)',
        'card_glucose_title': '🩸 Blood Sugar (Glucose)',
        'card_bp_title': '🫀 Blood Pressure',
        'card_hr_title': '❤️ Heart Rate (Pulse)',
        'action_plan': 'Action Plan:',

        # BMI statuses & advice
        'stat_underweight': 'Underweight',
        'plan_underweight': 'Consider a balanced diet plan to reach a healthy weight. Consult a nutritionist.',
        'stat_overweight': 'Overweight',
        'plan_overweight': 'A balanced diet and regular physical activity can help. Aim for 150 mins of exercise weekly.',
        'stat_obese': 'Obesity',
        'plan_obese': 'It is highly recommended to consult a healthcare provider for a weight management plan.',
        'stat_healthy_weight': 'Healthy Range',
        'plan_healthy_weight': 'Keep up the good work! Maintain a balanced diet and regular exercise.',

        # Glucose statuses & advice
        'stat_above_normal': 'Above Normal',
        'plan_above_normal': 'Reducing intake of refined sugars and carbs may help manage your levels. Monitor regularly.',
        'stat_below_normal': 'Below Normal',
        'plan_below_normal': 'Ensure you are eating regular, balanced meals. Don\'t skip meals.',
        'stat_normal_glucose': 'Normal',
        'plan_normal_glucose': 'Excellent! Keep maintaining your current diet and healthy habits.',

        # BP statuses & advice
        'stat_elevated_bp': 'Elevated',
        'plan_elevated_bp': 'Reducing sodium intake, regular exercise, and stress management can help lower your BP.',
        'stat_normal_bp': 'Normal',
        'plan_normal_bp': 'Great job! Continue to monitor and maintain a healthy lifestyle.',

        # Heart rate statuses & advice
        'stat_high_hr': 'High',
        'plan_high_hr': 'Cardiovascular exercise and stress management might be beneficial.',
        'stat_low_hr': 'Low',
        'plan_low_hr': 'This can be normal for athletes, but monitor for dizziness or fatigue.',
        'stat_normal_hr': 'Normal',
        'plan_normal_hr': 'Your resting heart rate is in a healthy range.',

        # What-If Simulator
        'sim_desc': 'Adjust the sliders below to see how changes in your health metrics impact your diabetes risk in real-time.',
        'sim_title': 'Simulated Risk Probability',
        'sim_above': 'Your simulated risk is above the threshold ({thresh:.0f}%). Try lowering BMI, Glucose, or Blood Pressure.',
        'sim_below': 'Your simulated risk is in the safe zone (below {thresh:.0f}%)!',
        'sim_error': 'Model not loaded for simulation.'
    },
    'id': {
        # App & Navigation
        'app_title': 'SugarSense',
        'lang_label': '🌐 Bahasa / Language',
        'nav_dashboard': 'Beranda',
        'nav_prediction': 'Prediksi',
        'nav_report': 'Laporan',
        'greet_morning': 'Selamat Pagi',
        'greet_afternoon': 'Selamat Siang',
        'greet_evening': 'Selamat Malam',
        'greet_name': 'Sahabat!',
        'btn_check_new': '+  Cek baru',
        'chip_features': 'Fitur model',
        'chip_threshold': 'Batas keputusan',
        'chip_accuracy': 'Akurasi model',
        'chip_dataset': 'Dataset',
        'chip_records': 'data',
        'card_glucose_trend': 'Tren Gula Darah per Usia',
        'card_prevalence': 'Prevalensi Diabetes',
        'card_recent': 'Riwayat Cek',
        'card_health_info': 'Informasi Kesehatan',
        'card_bmi_rate': 'Persentase Diabetes per Kategori BMI',
        'card_ranges': 'Rentang Sehat',
        'pill_dataset': 'Dataset',
        'pill_today': 'Info hari ini',
        'pill_last_input': 'Input terakhir',
        'pill_model': 'Model',
        'pill_guide': 'Panduan',
        'pill_tips': 'Tips',
        'pill_fields': '14 kolom',
        'pill_model_output': 'Output model',
        'pill_personalized': 'Personal',
        'pill_how_to_read': 'Cara baca',
        'pill_result': 'Hasil',
        'of_people': 'dari {n:,} orang',
        'no_checks_title': 'Belum ada pengecekan',
        'no_checks_sub': 'Lakukan prediksi pertama agar muncul di sini.',
        'latest_check': 'Pengecekan terakhir',
        'below_avg': 'Di bawah rata-rata',
        'above_avg': 'Di atas rata-rata',
        'avg_label': 'rata-rata',
        'step_1': '① Isi data',
        'step_2': '② Lihat prediksi',
        'step_3': '③ Baca laporan',
        'tips_title': 'Sebelum mulai',
        'tips_desc': 'Gunakan hasil ukur terbaru. Gula darah puasa serta tekanan darah dan nadi saat istirahat memberi hasil paling akurat.',
        'card_your_metrics': 'Data Kesehatan Anda',
        'card_pred_prob': 'Probabilitas Prediksi',
        'report_subtitle': 'Berikut ringkasan hasil prediksi risiko diabetes Anda.',
        'chip_risk': 'Risiko diabetes',
        'neg_tip1_title': 'Tetap aktif',
        'neg_tip1_sub': 'Usahakan olahraga sekitar 150 menit per minggu.',
        'neg_tip2_title': 'Cek rutin',
        'neg_tip2_sub': 'Pantau kesehatan secara berkala tiap tahun.',
        'pos_tip1_title': 'Konsultasi dokter',
        'pos_tip1_sub': 'Sebaiknya segera periksakan diri ke tenaga medis.',
        'pos_tip2_title': 'Evaluasi pola hidup',
        'pos_tip2_sub': 'Jaga pola makan, olahraga, dan pantau gula darah.',
        'kv_algorithm': 'Algoritma',
        'kv_features': 'Fitur input',
        'kv_features_val': '14 indikator',
        'kv_threshold': 'Batas risiko',
        'kv_accuracy': 'Akurasi uji',
        'kv_training': 'Data latih',
        'kv_output': 'Output',
        'range_glucose': 'Gula Darah',
        'range_bmi': 'Indeks Massa Tubuh',
        'range_bp': 'Tekanan Darah',
        'range_pulse': 'Denyut Nadi',
        
        # Dashboard / Home
        'hero_title': 'Selamat Datang!',
        'hero_subtitle': 'Cek risiko diabetes Anda sejak dini, dengan cara yang mudah dan cepat.',
        'what_is_title': 'Apa Itu SugarSense?',
        'what_is_desc': (
            'SugarSense adalah aplikasi web berbasis AI yang membantu memperkirakan risiko diabetes '
            'dari data kesehatan dasar, seperti kadar gula darah, BMI, dan tekanan darah. '
            'Hasilnya bisa jadi pengingat awal agar Anda lebih peduli pada kesehatan dan pola hidup.'
        ),
        'what_do_title': 'Apa yang Bisa Dilakukan SugarSense?',
        'what_do_desc': (
            'SugarSense memakai machine learning untuk membaca data kesehatan Anda '
            'dan memperkirakan risiko diabetes hanya dalam hitungan detik.'
        ),
        'why_title': 'Kenapa SugarSense?',
        'why_desc': (
            'Mengetahui risiko sejak dini sangat berarti. SugarSense dibuat supaya skrining kesehatan '
            'terasa lebih mudah, sederhana, dan bisa dipahami siapa saja.'
        ),
        'disclaimer_title': '⚠️ Hanya untuk Edukasi',
        'disclaimer_desc': (
            'SugarSense hanya untuk keperluan edukasi dan tidak menggantikan diagnosis atau saran dokter. '
            'Selalu konsultasikan kondisi kesehatan Anda ke tenaga medis.'
        ),
        'btn_start': '+ Mulai Prediksi Baru',

        # Input Page
        'form_title': 'Data Kesehatan',
        'form_subtitle': 'Isi data kesehatan Anda di bawah ini untuk mendapatkan hasil prediksi.',
        'label_age': 'Usia (tahun)',
        'help_age': 'Usia Anda saat ini',
        'label_gender': 'Jenis Kelamin',
        'opt_male': 'Laki-laki',
        'opt_female': 'Perempuan',
        'label_height': 'Tinggi Badan (cm)',
        'help_height': 'Isi dalam sentimeter, contoh: 160',
        'label_glucose': 'Kadar Gula Darah (mmol/L)',
        'help_glucose': 'Normalnya 4.0-7.0 mmol/L',
        'label_hypertensive': 'Hipertensi (Darah Tinggi)',
        'label_family_diabetes': 'Riwayat Diabetes di Keluarga',
        'help_family_diabetes': 'Apakah ada anggota keluarga dekat yang menderita diabetes?',
        'label_family_hypertension': 'Riwayat Hipertensi di Keluarga',
        'help_family_hypertension': 'Apakah ada anggota keluarga dekat yang menderita darah tinggi?',
        'label_cardiovascular': 'Penyakit Jantung & Pembuluh Darah',
        'help_cardiovascular': 'Pernah didiagnosis dokter menderita penyakit jantung atau pembuluh darah?',
        'label_stroke': 'Pernah Stroke',
        'help_stroke': 'Apakah Anda pernah terkena stroke?',
        'opt_no': 'Tidak',
        'opt_yes': 'Ya',
        'label_systolic': 'Tekanan Darah Atas / Sistolik (mmHg)',
        'help_systolic': 'Normalnya di bawah 120 mmHg',
        'label_diastolic': 'Tekanan Darah Bawah / Diastolik (mmHg)',
        'help_diastolic': 'Normalnya di bawah 80 mmHg',
        'label_bmi': 'Indeks Massa Tubuh (BMI)',
        'help_bmi': 'Normalnya 18.5-24.9',
        'label_weight': 'Berat Badan (kg)',
        'label_pulse': 'Denyut Nadi (bpm)',
        'help_pulse': 'Normalnya 60-100 bpm',
        'btn_predict': 'Lihat Hasil Prediksi',
        'err_zero': '⚠️ Mohon isi semua kolom dengan benar. Nilainya tidak boleh 0.',
        'err_model': '❌ File model tidak ditemukan. Pastikan model.pkl ada di folder aplikasi.',

        # Result Pages
        'report_heading': 'Hasil Prediksi',
        'neg_title': 'Tidak Terdeteksi Diabetes',
        'neg_subtitle': 'Dari data yang Anda isi, risiko diabetes Anda tergolong rendah.',
        'pos_title': 'Terdeteksi Berisiko Diabetes',
        'pos_subtitle': 'Model menemukan tanda-tanda yang sering dikaitkan dengan diabetes.',
        'chart_safe': 'Aman',
        'chart_risk': 'Risiko',
        'chart_non_diab': 'Bukan Diabetes',
        'chart_diab': 'Diabetes',
        'analysis_title': 'Ringkasan Kondisi Anda',
        'neg_analysis_desc': (
            'Berdasarkan data yang Anda isi, kemungkinan Anda menderita diabetes tergolong kecil. '
            'Grafik di samping menunjukkan seberapa yakin model terhadap hasil ini. '
            'Tetap jaga pola hidup sehat dan rawat tubuh Anda.'
        ),
        'neg_highlight': '💡 Tetap jaga pola hidup sehat dan rutin cek kesehatan.',
        'pos_analysis_desc': (
            'Berdasarkan data yang Anda isi, kemungkinan Anda menderita diabetes sekitar <b>{prob:.0f}%</b>. '
            'Grafik di samping menunjukkan seberapa yakin model terhadap hasil ini.<br><br>'
            'Tenang, ini baru perkiraan awal dari model dan bukan diagnosis. '
            'Anggap saja sebagai pengingat untuk lebih memperhatikan kesehatan.'
        ),
        'pos_highlight': '💡 Sebaiknya segera periksakan diri ke dokter.',
        'btn_new_pred': 'Prediksi Lagi',

        # Result explanation (what the numbers mean - short & plain)
        'explain_title': '📖 Cara Membaca Hasil Ini',
        'explain_intro': 'Penjelasan singkat arti setiap angka di atas.',
        'explain_score_title': 'Skor Risiko Anda',
        'explain_score_desc': 'Perkiraan risiko diabetes dari model AI. Makin tinggi berarti risikonya makin besar (bukan diagnosis dokter).',
        'explain_safe_note': 'Angka "Aman" di grafik adalah 100% dikurangi skor ini.',
        'explain_threshold_title': 'Batas Keputusan',
        'explain_threshold_desc': 'Garis batas penentu. Skor di atas garis ini ditandai berisiko diabetes.',
        'explain_threshold_above': 'Skor Anda ({score:.0f}%) di atas batas → {label}.',
        'explain_threshold_below': 'Skor Anda ({score:.0f}%) di bawah batas → {label}.',
        'explain_accuracy_title': 'Akurasi Model',
        'explain_accuracy_desc': 'Dari uji coba pada {n:,} orang, model menebak benar sekitar {correct:.0f} dari 100 orang.',
        'explain_note_title': 'Catatan singkat soal akurasi',
        'explain_note_desc': 'Di data latih hanya sekitar {share:.0f} dari 100 orang yang diabetes, jadi 2 angka ini juga penting:',
        'explain_recall': '<b>Recall {recall:.0f}%</b>: berhasil mendeteksi sekitar {recall:.0f} dari 100 kasus diabetes asli.',
        'explain_precision': '<b>Presisi {precision:.0f}%</b>: sekitar {precision:.0f} dari 100 hasil positif memang diabetes — anggap hasil positif sebagai pengingat untuk cek ke dokter.',

        # Interactive Insights
        'insights_title': '📌 Wawasan Kesehatan Anda',
        'tab_deep_dive': '🔍 Penjelasan Detail',
        'tab_gauges': '📊 Indikator Visual',
        'tab_simulator': '🎛️ Simulasi "Bagaimana Jika"',

        # Deep Dive Cards
        'card_bmi_title': '⚖️ Berat Badan (BMI)',
        'card_glucose_title': '🩸 Gula Darah',
        'card_bp_title': '🫀 Tekanan Darah',
        'card_hr_title': '❤️ Denyut Nadi',
        'action_plan': 'Saran:',

        # BMI statuses & advice
        'stat_underweight': 'Berat Kurang',
        'plan_underweight': 'Atur pola makan bergizi seimbang agar berat badan naik ke batas sehat. Sebaiknya konsultasi dengan ahli gizi.',
        'stat_overweight': 'Berat Berlebih',
        'plan_overweight': 'Pola makan seimbang dan rutin bergerak bisa membantu. Usahakan olahraga sekitar 150 menit per minggu.',
        'stat_obese': 'Obesitas',
        'plan_obese': 'Sebaiknya konsultasi dengan dokter untuk menyusun program penurunan berat badan.',
        'stat_healthy_weight': 'Normal',
        'plan_healthy_weight': 'Bagus! Pertahankan pola makan seimbang dan rajin berolahraga.',

        # Glucose statuses & advice
        'stat_above_normal': 'Di Atas Normal',
        'plan_above_normal': 'Coba kurangi makanan dan minuman manis serta karbohidrat olahan, dan cek gula darah secara rutin.',
        'stat_below_normal': 'Di Bawah Normal',
        'plan_below_normal': 'Makan teratur dengan menu seimbang dan jangan sampai melewatkan jam makan.',
        'stat_normal_glucose': 'Normal',
        'plan_normal_glucose': 'Bagus! Pertahankan pola makan dan kebiasaan sehat Anda.',

        # BP statuses & advice
        'stat_elevated_bp': 'Tinggi',
        'plan_elevated_bp': 'Kurangi garam, rutin berolahraga, dan kelola stres untuk membantu menurunkan tekanan darah.',
        'stat_normal_bp': 'Normal',
        'plan_normal_bp': 'Bagus! Terus pantau tekanan darah dan jaga pola hidup sehat.',

        # Heart rate statuses & advice
        'stat_high_hr': 'Tinggi',
        'plan_high_hr': 'Olahraga kardio dan mengelola stres bisa membantu menurunkan denyut nadi.',
        'stat_low_hr': 'Rendah',
        'plan_low_hr': 'Ini bisa normal pada orang yang rutin berolahraga, tetapi waspada jika Anda sering pusing atau mudah lemas.',
        'stat_normal_hr': 'Normal',
        'plan_normal_hr': 'Denyut nadi istirahat Anda masih dalam rentang sehat.',

        # What-If Simulator
        'sim_desc': 'Geser slider di bawah untuk melihat bagaimana perubahan data kesehatan memengaruhi perkiraan risiko diabetes Anda.',
        'sim_title': 'Perkiraan Risiko Hasil Simulasi',
        'sim_above': 'Risiko hasil simulasi masih di atas batas ({thresh:.0f}%). Coba turunkan BMI, gula darah, atau tekanan darah.',
        'sim_below': 'Risiko hasil simulasi sudah di bawah batas ({thresh:.0f}%). Aman!',
        'sim_error': 'Model tidak bisa dimuat untuk simulasi.'
    }
}

def t(key, lang='en', **kwargs):
    """Retrieve translated string with optional formatting arguments."""
    lang_dict = TRANSLATIONS.get(lang, TRANSLATIONS['en'])
    text = lang_dict.get(key, TRANSLATIONS['en'].get(key, key))
    if kwargs:
        return text.format(**kwargs)
    return text
