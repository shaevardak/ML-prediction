import joblib
import pandas as pd
import streamlit as st

from train import FEATURES

st.set_page_config(page_title="Прогноз возврата кредита", page_icon="💳", layout="wide")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


try:
    art = load()
except FileNotFoundError:
    st.error("Файл model.joblib не найден. Сначала запусти: "
             "`python train.py accepted_2007_to_2018Q2.csv`")
    st.stop()

pipe, opts, d = art["pipeline"], art["options"], art["defaults"]

st.title("💳 Прогноз возврата кредита")
st.caption("Модель на данных Lending Club: оценивает вероятность того, "
           "что заёмщик не вернёт кредит.")

tab_pred, tab_model = st.tabs(["Прогноз", "О модели"])

with tab_pred:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Кредит")
        loan_amnt = st.number_input("Сумма кредита, $", 500, 40000, 10000, 500)
        term = st.selectbox("Срок, мес.", [36, 60])
        int_rate = st.slider("Процентная ставка, %", 5.0, 31.0, 13.0, 0.1)
        installment = st.number_input("Ежемесячный платёж, $", 10.0, 1800.0,
                                      float(round(d["installment"])), 10.0)
        grade = st.selectbox("Рейтинг (grade)", opts["grade"],
                             index=min(2, len(opts["grade"]) - 1))
        purpose = st.selectbox("Цель кредита", opts["purpose"])
    with c2:
        st.subheader("Заёмщик")
        annual_inc = st.number_input("Годовой доход, $", 4000, 1_000_000, 65000, 1000)
        emp_length = st.slider("Стаж работы, лет", 0, 10, 5)
        home = st.selectbox("Жильё", opts["home_ownership"])
        verif = st.selectbox("Проверка дохода", opts["verification_status"])
        dti = st.slider("Долговая нагрузка (DTI), %", 0.0, 60.0, 18.0, 0.5)
        fico = st.slider("FICO score", 600, 850, 690, 5)
    with c3:
        st.subheader("Кредитная история")
        hist = st.slider("Длина истории, лет", 0.0, 50.0, 15.0, 0.5)
        open_acc = st.number_input("Открытых счетов", 0, 60, int(d["open_acc"]))
        total_acc = st.number_input("Всего счетов", 0, 120, int(d["total_acc"]))
        revol_bal = st.number_input("Револьверный долг, $", 0, 500000, int(d["revol_bal"]), 500)
        revol_util = st.slider("Использование лимита, %", 0.0, 120.0, float(d["revol_util"]), 0.5)
        delinq = st.number_input("Просрочек за 2 года", 0, 20, 0)
        pub_rec = st.number_input("Публичных записей", 0, 10, 0)
        bankr = st.number_input("Банкротств", 0, 5, 0)
        mort = st.number_input("Ипотечных счетов", 0, 20, int(d["mort_acc"]))

    if st.button("Рассчитать риск", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
            "installment": installment, "emp_length": emp_length,
            "annual_inc": annual_inc, "dti": dti, "delinq_2yrs": delinq,
            "fico": fico, "open_acc": open_acc, "pub_rec": pub_rec,
            "revol_bal": revol_bal, "revol_util": revol_util,
            "total_acc": total_acc, "mort_acc": mort,
            "pub_rec_bankruptcies": bankr, "credit_history_years": hist,
            "grade": grade, "home_ownership": home,
            "verification_status": verif, "purpose": purpose,
        }])[FEATURES]
        p = float(pipe.predict_proba(row)[0, 1])
        base = art["metrics"]["base_rate"]

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Вероятность дефолта", f"{p:.1%}")
        m2.metric("Средний уровень по выборке", f"{base:.1%}")
        level = "Низкий" if p < base * 0.7 else "Средний" if p < base * 1.3 else "Высокий"
        m3.metric("Уровень риска", level)
        st.progress(min(p, 1.0))
        if level == "Низкий":
            st.success("Заёмщик выглядит надёжнее среднего.")
        elif level == "Средний":
            st.warning("Риск близок к среднему по портфелю.")
        else:
            st.error("Риск выше среднего: нужна дополнительная проверка.")

with tab_model:
    m = art["metrics"]
    a, b, c = st.columns(3)
    a.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    b.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    c.metric("Доля дефолтов", f"{m['base_rate']:.1%}")
    st.subheader("Самые важные признаки")
    imp = pd.Series(art["importance"]).sort_values()
    st.bar_chart(imp, horizontal=True)
    st.caption("Важность считается методом permutation importance "
               "(падение ROC-AUC при перемешивании признака).")
    st.info("Это учебная модель на исторических данных. "
            "Не используйте её для реальных кредитных решений.")
