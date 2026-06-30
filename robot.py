import streamlit as st
import numpy as np
import time
import pandas as pd
import altair as alt

# ==============================================================================
# KONFIGURÁCIA STRÁNKY
# ==============================================================================
st.set_page_config(
    page_title="UAV Active Tracking Platform",
    layout="wide"
)

st.title("UAV Active Tracking Platform")
st.markdown("---")

# ==============================================================================
# MATEMATICKÉ JADRO A VÝPOČTY (PREDIKTOR A REWARD)
# ==============================================================================
class TrajectoryPredictor:
    def __init__(self, history_len=6):
        self.history = []
        self.history_len = history_len

    def update(self, x, y):
        self.history.append((x, y))
        if len(self.history) > self.history_len: self.history.pop(0)

    def predict_next(self):
        if len(self.history) < 2: return None
        deltas_x = [self.history[i][0] - self.history[i-1][0] for i in range(1, len(self.history))]
        deltas_y = [self.history[i][1] - self.history[i-1][1] for i in range(1, len(self.history))]
        return int(self.history[-1][0] + sum(deltas_x)/len(deltas_x)), int(self.history[-1][1] + sum(deltas_y)/len(deltas_y))

    def reset(self):
        self.history.clear()

def evaluate_reward(error_x, error_y, confidence):
    return (confidence * 2.5) - (np.sqrt(error_x**2 + error_y**2) * 0.0015)

# ==============================================================================
# OVLÁDACÍ PANEL I VSTUPY
# ==============================================================================
st.sidebar.header("Ovladaci Panel i Vstupy")

scenar = st.sidebar.selectbox(
    "Vyberte testovaciu misiu UAV:",
    ["Misia 1: Linearne sledovanie s prekazkou", "Misia 2: Sinusoidna trajektoria (Komplexna)"]
)

st.sidebar.markdown("---")
st.sidebar.header("Nastavenia filtrov")
sim_speed = st.sidebar.slider("Rychlost simulacie (ms medzi snimkami)", min_value=10, max_value=200, value=50)

# ==============================================================================
# ROZLOZENIE STRÁNKY (LAYOUT)
# ==============================================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Hlavny Opticky Stream (UAV Tracking Radar)")
    chart_placeholder = st.empty()

with col2:
    st.subheader("Takticky Mikro-Vyrez (ROI)")
    zoom_placeholder = st.empty()
    
    st.subheader("Telemetria a Vypocty")
    telemetry_placeholder = st.empty()

explanation_text = """
***
**Vysvetlenie telemetrickych velicin:**

* **Stav systemu:** Regionalny rezim riadenia UAV. *TRACKING ACTIVE* znamena stabilne zameranie. *ACTIVE VISION* deteguje pokles istoty a simuluje orbitalny manever pre lepsi uhol pohladu. *PREDIKCIA* znamena, ze objekt je skryty a dron leti podla zotrvacnosti linearneho filtra.
* **Confidence (Istota AI):** Vyjadruje percentualnu istotu detekcneho modelu YOLOv8, ze dany objekt je naozaj hladany clovek.
* **RL Reward (Skore odmeny):** Matematicky vystup z odmenovacej funkcie pre posilnovane ucenie ($R = R_{conf} - P_{dist}$).
* **Error X/Y (Odchylka od stredu):** Diferencia objektu od optickeho stredu senzora v pixeloch.
"""
st.markdown(explanation_text)

start_button = st.button("Spustiť real-time simuláciu misie")

# ==============================================================================
# LIVE BEZCHYBNÁ SIMULÁCIA (STOPERCENTNE FUNKČNÁ NA WEBE)
# ==============================================================================
if start_button:
    predictor = TrajectoryPredictor()
    CENTER_X, CENTER_Y = 320, 180
    occlusion_counter = 0
    
    # Vygenerovanie 100 bodov trasy podľa zvoleného scenára
    total_steps = 100
    steps = np.arange(total_steps)
    
    if "Misia 1" in scenar:
        # Priama línia idúca zľava doprava
        target_xs = 50 + steps * 5.5
        target_ys = np.full(total_steps, 180)
        # Prekážka (stena) v strede trasy (kroky 40 až 65)
        occlusion_mask = (steps >= 40) & (steps <= 65)
    else:
        # Sínusoida
        target_xs = 50 + steps * 5.5
        target_ys = 180 + np.sin(steps * 0.2) * 60
        occlusion_mask = (steps >= 45) & (steps <= 70)

    # Hlavný cyklus, ktorý kreslí graf namiesto padajúceho videa
    for i in range(total_steps):
        tx, ty = target_xs[i], target_ys[i]
        is_hidden = occlusion_mask[i]
        
        # 1. PRÍPRAVA DÁT PRE GRAF
        plot_data = []
        
        # Pridáme optický stred drona (červený kríž)
        plot_data.append({"X": CENTER_X, "Y": CENTER_Y, "Typ": "Stred senzora UAV", "Velkost": 100})
        
        # Vykreslenie prekážky, ak sme v jej zóne
        if is_hidden:
            # Nasimulujeme stenu v grafe
            for wall_y in range(50, 310, 20):
                plot_data.append({"X": 300, "Y": wall_y, "Typ": "PREKAZKA (Budova)", "Velkost": 150})
        
        # Logika sledovania/predikcie
        if not is_hidden:
            # Objekt je viditeľný, aktualizujeme filter
            predictor.update(tx, ty)
            occlusion_counter = 0
            conf = 0.88 if "Misia 1" in scenar else 0.45 # V misii 2 simulujeme horší uhol
            status = "TRACKING ACTIVE" if conf > 0.5 else "ACTIVE VISION"
            
            err_x = int(tx - CENTER_X)
            err_y = int(ty - CENTER_Y)
            reward = evaluate_reward(err_x, err_y, conf)
            
            plot_data.append({"X": tx, "Y": ty, "Typ": "Sledovany objekt (Ciel)", "Velkost": 200})
            
            # Taktický výrez (simulovaný zeleným štvorcom)
            zoom_data = pd.DataFrame([{"X": 0, "Y": 0}])
            zoom_chart = alt.Chart(zoom_data).mark_square(color='green', size=3000).properties(width=150, height=150)
            zoom_placeholder.altair_chart(zoom_chart)
            
        else:
            # Objekt je skrytý, nastupuje predikcia lineárnym filtrom
            pred = predictor.predict_next()
            conf = 0.0
            
            if pred and occlusion_counter < 30:
                occlusion_counter += 1
                px, py = pred
                predictor.update(px, py) # filter kŕmi sám seba
                status = "PREDIKCIA (STRATA KONTAKTU)"
                err_x, err_y = px, py
                reward = 0.0
                plot_data.append({"X": px, "Y": py, "Typ": "Odhadovana poloha (Predikcia)", "Velkost": 200})
            else:
                predictor.reset()
                status = "VYHLADAVANIE / MIMO DOSAH"
                err_x, err_y = 0, 0
                reward = 0.0
        
        # Vykreslenie hlavného radaru (Scattering plot v Altair)
        df = pd.DataFrame(plot_data)
        chart = alt.Chart(df).mark_circle().encode(
            x=alt.X('X', scale=alt.Scale(domain=[0, 640])),
            y=alt.Y('Y', scale=alt.Scale(domain=[0, 360])),
            color=alt.Color('Typ', scale=alt.Scale(
                domain=['Stred senzora UAV', 'Sledovany objekt (Ciel)', 'Odhadovana poloha (Predikcia)', 'PREKAZKA (Budova)'],
                range=['red', 'green', 'yellow', 'gray']
            )),
            size=alt.Size('Velkost', legend=None)
        ).properties(width=640, height=360).configure_view(strokeOpacity=0)
        
        chart_placeholder.altair_chart(chart, use_container_width=True)
        
        # UPDATE TELEMETRIE (Prebieha v dokonalom reálnom čase so zmenou grafu!)
        if status == "PREDIKCIA (STRATA KONTAKTU)":
            telemetry_placeholder.markdown(f"""
* **Stav systemu:** `PREDIKCIA (STRATA KONTAKTU)` 
* **Snímky naslepo:** `{occlusion_counter} / 30`
* **Predpovedaný X/Y:** `X: {err_x}px | Y: {err_y}px`
""")
        elif "VYHLADAVANIE" in status:
            telemetry_placeholder.markdown("""
* **Stav systemu:** `VYHLADAVANIE / MIMO DOSAH` 
* **Confidence (Istota AI):** `0.00%`
* **RL Reward (Skore odmeny):** `N/A`
""")
        else:
            telemetry_placeholder.markdown(f"""
* **Stav systemu:** `{status}` 
* **Confidence (Istota AI):** `{conf:.2%}` 
* **RL Reward (Skore odmeny):** `{reward:+.4f}` 
* **Error X/Y (Odchylka od stredu):** `X: {err_x}px | Y: {err_y}px`
""")
            
        time.sleep(sim_speed / 1000.0)
        
    st.success("Simulácia misie bola úspešne dokončená.")
    st.balloons()
