import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('UpdateResumeDataSet.csv')

df.head()

df.shape

df['Category'].value_counts()

plt.figure(figsize=(15,5))
sns.countplot(df['Category'])
plt.xticks(rotation=90)
plt.show()

df['Category'].unique()

counts = df['Category'].value_counts()
labels =df['Category'].unique()
plt.figure(figsize=(15,10))

plt.pie(counts,labels=labela,autopct='%1.1f%%', shadow=True,colors=plt.cm.plasma(np.linspace(0,1,3)))

df['Category'][0]

df['Resume'][0]

#Balance Classes

print("Original Category Di3stribution:")
print(df['Category'].value_counts())


max_size =df['Category'].value_counts().max()

balanced_df = df.groupby('Category').apply(lambda x: x.sample(max_size,replace=True)).reset_index(drop=True)

df = balanced_df.sample(face=1).reset_index(drop=True)


print("\n Balaced Category Distribution(After Oversampling):")
print(df['Category'].value_counts())

#Cleaninig Data

import re
def cleanResume(txt):
    cleanText = re.sub('http\S+\S','',txt)
    cleanText = re.sub('RT|cc','',cleanText)
    cleanText = re.sub('#\s+\s','',cleanText)
    cleanText = re.sub('@\s+',' ',cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~3"""),' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]',' ',cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

cleanResume("my #### $ # #noorsaeed website like is this http://helword and acces it @gmail.com")

df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

df['Resume'][0]

#Word into categorical values
from sklearn.preproceesing import LabelEncoder
le = LabelEncoder()

le.fit(df['Category'])
df['Category']=le.transform(df['Category'])

df.Category.unique()

#Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = 'english')

tfidf.fit(df['Resume'])
requredTaxt = tfidf.transform(df['Resume'])

from sklearn.model_selection import train _test_split
X_train, X_test, y_train, y_test = train_test_split(requredTaxt,df['Category'],test_size=0.2,random_state=42)

X_train.shape

X_test.shape

#Train the model and print the classification report
from sklearn.neghbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegreesion
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Ensure that X_train and X_test are dense if they are sparse
X_train = X_tain.toarray() if hasattr(X_train,'toarray') else x_train
X_test = X_test.toarray() if hasattr(X_test,'toarray') else x_test

#Train KNeighborsClassifier
knn_model = OneVsRestClassifier(KNeighborsClassifier())
knn_model.fit(X_train,y_train)
y_pred_knn = knn_model.predict(X_test)
print("\nKNeighborsClassifier Results:")
print(f"Accurancy:{accuracy_score(y_test,y_pred_knn):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test,y_pred_knn)}")
print(f"Classification Report:\n{classification_report(y_test,y_pred_knn)}")

#Train SVC
svc_model = OneVsRestClassifier(SVC())
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(x_test)
print("\nSVC Results:")
print(f"Accuracy:{accuracy_score(y_test,y_pred_svc):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svc)}")
print(f"Classification Report:\n{classification_report(y_test,y_pred_svc)}")

#train RandomForestClassifier
rf_model = oneVsRestClassifier(RandomForestClassifier())
rf_model.fit(x_train, y_tarin)
y_pred_rf= rf_model.predict(X_test)
print("\nRandomForestClassifier Results:")
print(f"Accuracy:{accuracy_score(y_test,y_pred_rf):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test,y_pred_rf)}")
print(f"Classification Report:\n{classification_report(y_test,y_pred_rf)}")

import pickle
pickle.dump(tfidf,open('tfidf.pkl','wb'))
pickle.dump(svc_model,open('clf.pkl','wb'))
pickle.dump(le,open("encoder.pkl",'wb'))

#Prediction System
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)

    vectorized_text = tfidf.transform([cleaned_text])

    vectorized_text = vectorized_text.toaaray()

    predicted_category =svc_model.predict(vectorized_text)

    predicted_category_name =le.inverse_transform(predicted_category)

    return predicted_category_name[0]


myresume = """I am a data scientist specializing in machine
learning, deep learning, and computer vision. With
a strong background in mathematics, statistics,
and programming, I am passionate about
uncovering hidden patterns and insights in data.
I have extensive experience in developing
predictive models, implementing deep learning
algorithms, and designing computer vision
systems. My technical skills include proficiency in
Python, Sklearn, TensorFlow, and PyTorch.
What sets me apart is my ability to effectively
communicate complex concepts to diverse
audiences. I excel in translating technical insights
into actionable recommendations that drive
informed decision-making.
If you're looking for a dedicated and versatile data
scientist to collaborate on impactful projects, I am
eager to contribute my expertise. Let's harness the
power of data together to unlock new possibilities
and shape a better future.
Contact & Sources
Email: 611noorsaeed@gmail.com
Phone: 03442826192
Github: https://github.com/611noorsaeed
Linkdin: https://www.linkedin.com/in/noor-saeed654a23263/
Blogs: https://medium.com/@611noorsaeed
Youtube: Artificial Intelligence
ABOUT ME
WORK EXPERIENCE
SKILLES
NOOR SAEED
LANGUAGES
English
Urdu
Hindi
I am a versatile data scientist with expertise in a wide
range of projects, including machine learning,
recommendation systems, deep learning, and computer
vision. Throughout my career, I have successfully
developed and deployed various machine learning models
to solve complex problems and drive data-driven
decision-making
Machine Learnine
Deep Learning
Computer Vision
Recommendation Systems
Data Visualization
Programming Languages (Python, SQL)
Data Preprocessing and Feature Engineering
Model Evaluation and Deployment
Statistical Analysis
Communication and Collaboration
"""

pred(myresume)

myresume = """
Jane Smith is a certified personal trainer with over 5 years of experience in helping individuals achieve their fitness goals. Specializing in weight loss, strength training, and sports conditioning, Jane has developed personalized workout routines for clients of all ages and fitness levels. She has extensive knowledge in nutrition and exercise science, and uses this to create holistic health and fitness programs that are tailored to individual needs.

Jane holds a degree in Exercise Science and is a certified trainer through the National Academy of Sports Medicine (NASM). She has worked with athletes, seniors, and individuals with chronic health conditions, helping them improve their physical well-being and overall quality of life.

Her expertise includes:
- Weight Loss and Body Composition
- Strength Training and Resistance Exercises
- Cardio Conditioning
- Nutrition Coaching and Meal Planning
- Injury Prevention and Rehabilitation
- Functional Movement and Flexibility Training
- Group Fitness Classes

Certifications:
- Certified Personal Trainer, NASM
- CPR and First Aid Certified
- Yoga Instructor (200-Hour Certification)

Education:
BSc in Exercise Science, ABC University, 2014-2018

Work Experience:
- Personal Trainer at XYZ Fitness Gym (2018-Present)
- Fitness Coach at Wellness Center (2016-2018)

Languages:
- English (Fluent)
- Spanish (Conversational)
"""

pred(myresume)

myresume = """
John Doe is an experienced Network Security Engineer with over 7 years of expertise in designing, implementing, and managing network security infrastructures. Specializing in safeguarding critical network systems, John has worked with various organizations to protect against cyber threats, data breaches, and unauthorized access. He is proficient in deploying firewalls, intrusion detection systems (IDS), VPNs, and network monitoring tools to ensure the integrity and security of networks.

John holds a degree in Computer Science and certifications in several cybersecurity domains, including Certified Information Systems Security Professional (CISSP), Certified Ethical Hacker (CEH), and Cisco Certified Network Associate (CCNA). He has extensive experience in troubleshooting and resolving network vulnerabilities, and has played a key role in conducting security audits and risk assessments.

Key Skills:
- Network Security Architecture
- Firewall Management and Configuration
- Intrusion Detection and Prevention Systems (IDS/IPS)
- Virtual Private Networks (VPNs)
- Security Audits and Risk Assessments
- Cybersecurity Incident Response
- Network Monitoring and Traffic Analysis
- Vulnerability Assessment and Penetration Testing
- Data Encryption and Secure Communications

Certifications:
- CISSP (Certified Information Systems Security Professional)
- CEH (Certified Ethical Hacker)
- CCNA (Cisco Certified Network Associate)
- CompTIA Security+

Education:
BSc in Computer Science, XYZ University, 2012-2016

Professional Experience:
- Network Security Engineer at ABC Corp (2016-Present)
- IT Security Specialist at DEF Solutions (2014-2016)

Languages:
- English (Fluent)
- French (Intermediate)
"""


pred(myresume)

myresume = """
Sarah Williams is a dedicated and skilled advocate with over 10 years of experience in providing legal representation to clients across various sectors, including criminal law, civil litigation, and family law. With a deep understanding of legal procedures and case law, Sarah has successfully handled numerous cases in the courtroom, negotiating favorable settlements and providing expert legal advice to individuals and businesses.

She holds a law degree from XYZ University and is a licensed attorney, practicing law in multiple jurisdictions. Sarah is passionate about ensuring justice is served and strives to make legal processes more accessible to her clients. She is known for her excellent research and analytical skills, attention to detail, and commitment to upholding the law with integrity.

Key Skills:
- Criminal Law
- Civil Litigation
- Family Law
- Contract Law
- Legal Research and Writing
- Courtroom Advocacy
- Legal Counseling and Advice
- Client Relationship Management
- Legal Compliance and Regulations
- Negotiation and Settlement

Certifications and Licenses:
- Licensed Attorney at Law, XYZ State Bar
- Certification in Criminal Law, XYZ University

Education:
JD in Law, XYZ University, 2010-2013

Professional Experience:
- Senior Advocate at ABC Law Firm (2016-Present)
- Associate Advocate at DEF Legal Group (2013-2016)

Languages:
- English (Fluent)
- Spanish (Conversational)
"""
pred(myresume)
