import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

##Opening
st.markdown("<h1 style='text-align: center; color: white;'>Analysis and Classification of DASS Quiz Participants Response to Level of Depression, Stress and Anxiety</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify; color: white;'>Kesehatan jiwa manusia masih menjadi permasalahan utama di bidang kesehatan yang sangat signifikan di dunia. Dikarenakan maraknya penderita yang mengalami gangguan pada kesehatan jiwa seseorang, maka disini akan dilakukan analisis untuk mengetahui bagaimana perkembangan dari masalah tersebut. Disini akan digunakan data hasil survey yang diperoleh secara online dari Depression Anxiety Stress Scale (DASS), pada sebuah website yang terbuka untuk umum dari tahun 2017 hingga 2019. Dari hasil survey tersebut akan dilakukan proses klasifikasi untuk mengelompokkan masing-masing gangguan kesehatan jiwa berdasarkan kategori-kategori yang ada pada survey. Untuk metode klasifikasi yang digunakan disini adalah Decision Tree, K-Nearest Neighbors, dan Multinomial Naive Bayes. Alasan penggunaan tiga metode tersebut adalah untuk dijadikan perbandingan pada tingkat akurasi nantinya. Tahapan-tahapan yang dilakukan adalah sebagai berikut.</p>", unsafe_allow_html=True)
st.write("1. Import Dataset")
st.write("2. Data Description")
st.write("3. Exploratory Data Analysis")
st.write("4. Pre-processing")
st.write("5. Modeling")
st.write("6. Demo App")

# Import Dataset
df_data = pd.read_csv("data.csv", sep=r'\t')

##Displaying data
st.write("## Dataset")
st.markdown("<p style='text-align: justify; color: white;'>Berikut adalah 20 baris pertama dari dataset yang akan digunakan.</p>", unsafe_allow_html=True)
st.dataframe(df_data.head(20))

##Data Description
st.write("## Data Description")
st.markdown("<p style='text-align: justify; color: white;'>Dataset ini merupakan respon subyek penelitian terhadap depresi dan kecemasan. Data yang dimasukkan merupakan data yang dihimpun pada tahun 2017-2019.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify; color: white;'>Fitur yang terdapat di dalamnya adalah:</p>", unsafe_allow_html=True)
st.write("1. Q(n)A --> Respon atas pertanyaan yang diberikan")
st.write("2. Q(n)E --> Kecepatan responden dalam merespon")
st.write("3. Q(n)I --> Posisi pertanyaan")
st.write("4. TIPI(n) --> Ten Items Personality Inventory")
st.write("5. VCL(n) --> Respon terkait gambar pada pertanyaan yang diberikan")
st.write("6. education")
st.write("7. urban --> Tipe daerah responden tinggal")
st.write("8. gender")
st.write("9. engnat --> Pertanyaan terkait apakah Bahasa Inggris merupakan bahasa nasionalnya")
st.write("10. age")
st.write("11. hand --> Tangan dominan")
st.write("12. religion")
st.write("13. orientation --> Orientasi sexual")
st.write("14. race")
st.write("15. voted --> Pertanyaan terkait keikutsertaan dalam eleksi")
st.write("16. married")
st.write("17. familysize")
st.write("18. major")

# Get the Question Number each Category
DASS_number_q = {'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
                 'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
                 'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]}

# Split the Question
dep = []
for i in DASS_number_q["Depression"]:
    dep.append('Q'+str(i)+'A')

stress = []
for i in DASS_number_q["Stress"]:
    stress.append('Q'+str(i)+'A')

anx = []
for i in DASS_number_q["Anxiety"]:
    anx.append('Q'+str(i)+'A')

# Filter Question
only_q = df_data.filter(regex='Q\d{1,2}A')

depression_q = only_q.filter(dep)
stress_q = only_q.filter(stress)
anxiety_q = only_q.filter(anx)

##EDA
st.write("## Exploratory Data Analysis")
st.markdown("<p style='text-align: justify; color: white;'>Score of Response</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify; color: white;'>1 = Did not apply to me at all</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify; color: white;'>2 = Applied to me to some degree, or some of the time</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify; color: white;'>3 = Applied to me to a considerable degree, or a good part of the time</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify; color: white;'>4 = Applied to me very much, or most of the time</p>", unsafe_allow_html=True)

st.write("### Depression")

bar_dep = [(depression_q[depression_q == i].sum(axis=1).sum())/(i) for i in range(1,5)]
fig_dep,ax_dep = plt.subplots()
ax_dep=sns.barplot(x = [1,2,3,4], y=bar_dep)
ax_dep.set_xlabel('Score')
ax_dep.set_title('Depression Scores')

sns.despine()
st.pyplot(fig_dep)
st.markdown("<p style='text-align: justify; color: white;'>Dari hasil visualisasi pada hasil respon kategori pertanyaan depression dapat diketahui bahwa response pada opsi kedua (Applied to me to some degree, or some of the time) adalah yang terbanyak namun persebaran datanya cukup merata.</p>", unsafe_allow_html=True)

st.write("### Stress")

bar_str = [(stress_q[stress_q == i].sum(axis=1).sum())/(i) for i in range(1,5)]
fig_str,ax_str = plt.subplots()
ax_str=sns.barplot(x = [1,2,3,4], y=bar_str)
ax_str.set_xlabel('Score')
ax_str.set_title('Stress Scores')

sns.despine()
st.pyplot(fig_str)
st.markdown("<p style='text-align: justify; color: white;'>Dari hasil visualisasi pada hasil respon kategori pertanyaan stress dapat diketahui bahwa response pada opsi kedua (Applied to me to some degree, or some of the time) juga terbanyak.</p>", unsafe_allow_html=True)

st.write("### Anxiety")

bar_anx = [(anxiety_q[anxiety_q == i].sum(axis=1).sum())/(i) for i in range(1,5)]
fig_anx,ax_anx = plt.subplots()
ax_anx=sns.barplot(x = [1,2,3,4], y=bar_anx)
ax_anx.set_xlabel('Score')
ax_anx.set_title('Anxiety Scores')

sns.despine()
st.pyplot(fig_anx)
st.markdown("<p style='text-align: justify; color: white;'>Dari hasil visualisasi pada hasil respon kategori pertanyaan anxiety dapat diketahui bahwa response pada opsi pertama (Did not apply to me at all) lebih dominan daripada respon pada opsi yang lain.</p>", unsafe_allow_html=True)

##Pre-Processing
st.write("## Pre-Processing")

def sub(data):
    return data.subtract(1, axis=1)

depression_q = sub(depression_q)
stress_q = sub(stress_q)
anxiety_q = sub(anxiety_q)

st.write("### Scoring")
st.markdown("<p style='text-align: justify; color: white;'>Menjumlahkan score yang diperoleh tiap responden.</p>", unsafe_allow_html=True)

# Scoring
def scores(data):
    col = list(data)
    data["scores"] = data[col].sum(axis=1)
    return data

train_dep = scores(depression_q)
train_str = scores(stress_q)
train_anx = scores(anxiety_q)

st.write("1. Depression")
st.dataframe(train_dep.head())
st.write("2. Stress")
st.dataframe(train_str.head())
st.write("3. Anxiety")
st.dataframe(train_anx.head())

st.write("### Give a Label")
st.markdown("<p style='text-align: justify; color: white;'>Memberi label sesuai dengan total score yang diperoleh.</p>", unsafe_allow_html=True)

# Label Scale
DASS_scores = {'Depression': [(0, 10), (10, 14), (14, 21), (21, 28)],
               'Anxiety': [(0, 8), (8, 10), (10, 15), (15, 20)],
               'Stress': [(0, 15), (15, 19), (19, 26), (26, 34)]}

def label(data, string):
    conditions = [
    ((data['scores'] >= DASS_scores[string][0][0])  & (data['scores'] < DASS_scores[string][0][1])),
    ((data['scores'] >= DASS_scores[string][1][0])  & (data['scores'] < DASS_scores[string][1][1])),
    ((data['scores'] >= DASS_scores[string][2][0])  & (data['scores'] < DASS_scores[string][2][1])),
    ((data['scores'] >= DASS_scores[string][3][0])  & (data['scores'] < DASS_scores[string][3][1])),
    (((data['scores'] >= DASS_scores[string][3][1])))
    ]
    values = ['Normal','Mild', 'Moderate', 'Severe', 'Extremely Severe']
    data['category'] = np.select(conditions, values)
    return data
    
train_dep = label(train_dep, 'Depression')
train_str = label(train_str, "Stress")
train_anx = label(train_anx, "Anxiety")

st.write("1. Depression")
st.dataframe(train_dep.head())
st.write("2. Stress")
st.dataframe(train_str.head())
st.write("3. Anxiety")
st.dataframe(train_anx.head())

# Make Specific Label
train_dep["category"] = train_dep["category"].apply(lambda x: f"Depression {x}")
train_anx["category"] = train_anx["category"].apply(lambda x: f"Anxious {x}")
train_str["category"] = train_str["category"].apply(lambda x: f"Stress {x}")

st.write("### Value Counts")
st.markdown("<p style='text-align: justify; color: white;'>Menghitung total nilai masing-masing kategori.</p>", unsafe_allow_html=True)
st.write("1. Depression")
st.write(train_dep["category"].value_counts())
st.write("2. Stress")
st.write(train_str["category"].value_counts())
st.write("3. Anxiety")
st.write(train_anx["category"].value_counts())

##Pre-Processing
st.write("## Modeling")
st.markdown("<p style='text-align: justify; color: white;'>Metode Machine Learning yang dipilih adalah klasifikasi dengan modelnya yaitu Decision Tree.</p>", unsafe_allow_html=True)

# Data Training and Testing
X_dep = train_dep.drop(["category", "scores"], axis=1)
y_dep = train_dep["category"]

X_str = train_str.drop(["category", "scores"], axis=1)
y_str = train_str["category"]

X_anx = train_anx.drop(["category", "scores"], axis=1)
y_anx = train_anx["category"]

# Split Data
X_dep_train, X_dep_test, y_dep_train, y_dep_test = train_test_split(X_dep, y_dep, test_size=0.2, stratify=y_dep)
X_str_train, X_str_test, y_str_train, y_str_test = train_test_split(X_str, y_str, test_size=0.2, stratify=y_str)
X_anx_train, X_anx_test, y_anx_train, y_anx_test = train_test_split(X_anx, y_anx, test_size=0.2, stratify=y_anx)

##Demo
st.write("## Demo App")
st.markdown("<p style='text-align: justify; color: white;'>Pada bagian ini kita dapat mengisi survey dan memperoleh hasil prediksi terhadap level depresi, stres dan kecemasan yang kita miliki. Dalam mengisi survey ini terdapat tiga jenis kategori pertanyaan yaitu mengenai Depresi, Stres, dan Kecemasan yang dicampur secara acak.</p>", unsafe_allow_html=True)

def anxious_columns():
    return ['Q2A',
            'Q4A',
            'Q7A',
            'Q9A',
            'Q15A',
            'Q19A',
            'Q20A',
            'Q23A',
            'Q25A',
            'Q28A',
            'Q30A',
            'Q36A',
            'Q40A',
            'Q41A']

def stress_columns():
    return ['Q1A',
            'Q6A',
            'Q8A',
            'Q11A',
            'Q12A',
            'Q14A',
            'Q18A',
            'Q22A',
            'Q27A',
            'Q29A',
            'Q32A',
            'Q33A',
            'Q35A',
            'Q39A']

def depression_columns():
    return ['Q3A',
            'Q5A',
            'Q10A',
            'Q13A',
            'Q16A',
            'Q17A',
            'Q21A',
            'Q24A',
            'Q26A',
            'Q31A',
            'Q34A',
            'Q37A',
            'Q38A',
            'Q42A']

def questions():
    return ['I found myself getting upset by quite trivial things.',
            'I was aware of dryness of my mouth.',
            'I couldn\'t seem to experience any positive feeling at all.',
            'I experienced breathing difficulty (eg, excessively rapid breathing, breathlessness in the absence of physical exertion).',
            'I just couldn\'t seem to get going.',
            'I tended to over-react to situations.',
            'I had a feeling of shakiness (eg, legs going to give way).',
            'I found it difficult to relax.',
            'I found myself in situations that made me so anxious I was most relieved when they ended.',
            'I felt that I had nothing to look forward to.',
            'I found myself getting upset rather easily.',
            'I felt that I was using a lot of nervous energy.',
            'I felt sad and depressed.',
            'I found myself getting impatient when I was delayed in any way (eg, elevators, traffic lights, being kept waiting).',
            'I had a feeling of faintness.',
            'I felt that I had lost interest in just about everything.',
            'I felt I wasn&#39;t worth much as a person.',
            'I felt that I was rather touchy.',
            'I perspired noticeably (eg, hands sweaty) in the absence of high temperatures or physical exertion.',
            'I felt scared without any good reason.',
            'I felt that life wasn&#39;t worthwhile.',
            'I found it hard to wind down.',
            'I had difficulty in swallowing.',
            'I couldn&#39;t seem to get any enjoyment out of the things I did.',
            'I was aware of the action of my heart in the absence of physical exertion (eg, sense of heart rate increase, heart missing a beat).',
            'I felt down-hearted and blue.',
            'I found that I was very irritable.',
            'I felt I was close to panic.',
            'I found it hard to calm down after something upset me.',
            'I feared that I would be &quot;thrown&quot; by some trivial but unfamiliar task.',
            'I was unable to become enthusiastic about anything.',
            'I found it difficult to tolerate interruptions to what I was doing.',
            'I was in a state of nervous tension.',
            'I felt I was pretty worthless.',
            'I was intolerant of anything that kept me from getting on with what I was doing.',
            'I felt terrified.',
            'I could see nothing in the future to be hopeful about.',
            'I felt that life was meaningless.',
            'I found myself getting agitated.',
            'I was worried about situations in which I might panic and make a fool of myself.',
            'I experienced trembling (eg, in the hands).',
            'I found it difficult to work up the initiative to do things.']

def check_category(i):
    DASS_keys = {'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
                'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
                'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]}
    if i+1 in DASS_keys["Depression"]:
        return "Depression"
    elif i+1 in DASS_keys["Anxiety"]:
        return "Anxiety"
    elif i+1 in DASS_keys["Stress"]:
        return "Stress"

Q = questions()

anx_columns = anxious_columns()
dep_columns = depression_columns()
str_columns = stress_columns()

A = {'Did not apply to me at all' : 0,
     'Applied to me to some degree, or some of the time' : 1,
     'Applied to me to a considerable degree, or a good part of the time': 2,
     'Applied to me very much, or most of the time' : 3}

tmp_depression = []
tmp_anxiety = []
tmp_stress = []

tmp = []

with st.container():
    st.write("### Question of Depression, Stress and Anxiety")
    for i in range(0, len(Q), 2):
        col1, col2 = st.columns(2)
        with col1:
            answer_col_1 = st.radio(f"{Q[i]}",(A))
            category = check_category(i)
            if category == "Depression":
                tmp_depression.append(A[answer_col_1])
            if category == "Anxiety":
                tmp_anxiety.append(A[answer_col_1])
            if category == "Stress":
                tmp_stress.append(A[answer_col_1])
        with col2:
            answer_col_2 = st.radio(f"{Q[i+1]}",(A))
            category = check_category(i+1)
            if category == "Depression":
                tmp_depression.append(A[answer_col_2])
            if category == "Anxiety":
                tmp_anxiety.append(A[answer_col_2])
            if category == "Stress":
                tmp_stress.append(A[answer_col_2])

clf_dep = DecisionTreeClassifier()
clf_str = DecisionTreeClassifier()
clf_anx = DecisionTreeClassifier()

clf_dep.fit(X_dep_train, y_dep_train)
clf_str.fit(X_str_train, y_str_train)
clf_anx.fit(X_anx_train, y_anx_train)

result_depression = clf_dep.predict(pd.DataFrame({y : x for (x,y) in zip(tmp_depression, dep_columns)}, index=[0]))
result_stress = clf_str.predict(pd.DataFrame({y : x for (x,y) in zip(tmp_stress, str_columns)}, index=[0]))
result_anxious = clf_anx.predict(pd.DataFrame({y : x for (x,y) in zip(tmp_anxiety, anx_columns)}, index=[0]))

st.write("### Prediction Result")
st.markdown("<p style='text-align: justify; color: white;'>Berikut hasil prediksi terhadap level depresi, stres dan kecemasan yang kalian miliki berdasarkan hasil survey sebelumnya.</p>", unsafe_allow_html=True)
st.write(f"Depression level → {result_depression[0]}")
st.write(f"Stress level     → {result_stress[0]}")
st.write(f"Anxiety level    → {result_anxious[0]}")