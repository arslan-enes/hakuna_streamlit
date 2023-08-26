import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide")

@st.cache_data
def get_data():
    df = pd.read_csv("penguins.csv")
    return df

@st.cache_data
def get_pipeline():
    pipeline = joblib.load("penguin_pipeline.joblib")
    return pipeline

st.title(":blue[Penguin] :red[Classifier] 🐧🐧🐧")

main_page, data_page, model_page = st.tabs(["Ana Sayfa", "Veri Seti", "Model"])

# main page

information_container = main_page.container()
information_container.image("penguins.jpg", use_column_width=True)
information_container.subheader("Penguenler")
information_container.markdown("""
Penguenler, soğuk denizlerin ve buzulların sakinleri olarak adeta hayranlık uyandıran yaratıklardır. Karasal yaşamın en uç noktalarında, özellikle Antarktika ve çevresinde, farklı türleri bulunur. Bu türler arasında Adelie, Chinstrap ve Gentoo gibi öne çıkanları vardır. Penguenler, uçma yeteneklerinden yoksun olmalarına rağmen su altında ustaca yüzme kabiliyetine sahiptirler ve genellikle denizdeki avlarını kril, balık ve diğer deniz canlılarıyla sağlarlar. Beyaz, siyah ve bazen renkli lekelerle süslenmiş tüyleri, her bir türü benzersiz kılar. Penguenler aynı zamanda sosyal ve topluluk odaklı hayvanlardır, büyük kolonilerde yaşarlar ve üreme alanlarını korumak için çeşitli stratejiler geliştirirler. Hem karada hem de sularda gösterdikleri olağanüstü adaptasyonlar, penguenleri kutup bölgelerinin unutulmaz simgeleri haline getirir.

**Adelie Penguenleri**:

Adelie penguenleri, Antarktika'nın çevresindeki adalarda yaşayan orta büyüklükteki penguen türlerindendir. Siyah sırtları ve beyaz göğüsleri ile tanınırlar. Yüzgeçleri yüzme için uygundurken, karada yürümekte de oldukça ustadırlar. Özellikle denizdeki krillerle beslenirler ve yiyecek aramak için uzun mesafeleri kat edebilirler. Sosyal hayvanlardır ve büyük koloniler halinde yaşarlar. Yumurtalarını taşlı bölgelere veya kar üzerine bırakarak kuluçkaya yatarlar.

**Chinstrap Penguenleri**:

Chinstrap penguenleri, güney okyanus bölgelerinde yaşayan küçük penguen türlerindendir. Adını, çenesini altında bir çene kayışı gibi saran siyah bir banttan alırlar. Bu bant, beyaz yüzleriyle güzel bir tezat oluşturur. Kayalık bölgelerdeki kolonilerde yuvalarını yaparlar ve genellikle kril ve küçük balıklarla beslenirler. Hem denizde hem de karada hızlı hareket edebilme yetenekleri vardır.

**Gentoo Penguenleri**:

Gentoo penguenleri, Antarktika ve çevresindeki adalarda yaşayan bir penguen türüdür. Büyük boyutları ve beyaz leke benekli siyah sırtları ile tanınırlar. Başlarının üstünde belirgin bir beyaz şerit bulunur. Yüzme konusunda yeteneklidirler ve suda hızlı hareket edebilirler. Aynı zamanda iyi birer dalgıçtırlar ve çoğunlukla balıklar ve kril ile beslenirler. Yuvalarını genellikle bitki örtüsü veya taşlar arasına yaparlar.
""")

# data page

df = get_data()
data_page.dataframe(df, use_container_width=True)
data_page.divider()

data_page_col1, data_page_col2 = data_page.columns(2)

fig = plt.figure(figsize=(6,4))
sns.countplot(data=df, x="species")
data_page_col1.subheader("Türlerine Göre Penguen Sayısı")
data_page_col1.pyplot(fig)

fig2 = plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x="bill_length_mm", y="flipper_length_mm", hue="species")
data_page_col2.subheader("Penguenlerin Kanat ve Gaga Uzunlukları    ")
data_page_col2.pyplot(fig2)

# model page

pipeline = get_pipeline()

# model_page.write(pipeline.feature_names_in_)

# island
# bill_length_mm
# bill_depth_mm
# flipper_length_mm
# body_mass_g
# sex

## user inputs

user_input_col1, user_input_col2, result_col = model_page.columns([1, 1, 2])

user_island = user_input_col1.selectbox(label="Island", options=["Torgersen", "Dream", "Biscoe"])
user_bill_length_mm = user_input_col1.slider(label="Bill Length (mm)", min_value=20., max_value=80., step=0.5)
user_bill_depth_mm = user_input_col1.slider(label="Bill Depth (mm)", min_value=10., max_value=25., step=0.5)

user_flipper_length_mm = user_input_col2.number_input(label="Flipper Length (mm)", min_value=150., max_value=250., value=200., step=10.)
user_body_mass_g = user_input_col2.number_input(label="Body Mass (g)", min_value=2500, max_value=7000, value=4000, step=100)
user_sex = user_input_col2.radio(label="Sex", options=["Male", "Female"])

## prediction
user_input = pd.DataFrame({"island": user_island,
                           "bill_length_mm": user_bill_length_mm,
                           "bill_depth_mm": user_bill_depth_mm,
                           "flipper_length_mm": user_flipper_length_mm,
                           "body_mass_g": user_body_mass_g,
                           "sex": user_sex}, index=[0])


pictures = {"Adelie": "penguin_pics/adelie.jpeg",
            "Gentoo": "penguin_pics/gentoo.jpg",
            "Chinstrap": "penguin_pics/chinstrap.jpeg"}

if user_input_col2.button("Predict!"):
    result = pipeline.predict(user_input)[0]
    result_col.header(f"It is a/an {result}!", anchor=False)
    result_col.image(pictures[result], use_column_width=True)
    st.snow()


