import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.optimize
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
import os

app = Flask(__name__)

# Lig -> CSV dosyası eşleştirmesi
league_files = {
    "İngiltere": "england.csv",
    "Türkiye": "turkey.csv",
    "Fransa": "france.csv",
    "Almanya": "germany.csv",
    "İspanya": "spain.csv",
    "İtalya": "italy.csv"
}

#############################################################################
# (Opsiyonel) Uygulama başlarken tüm liglerin takımlarını yükleyip sözlük tutalım
#############################################################################
league_to_teams = {}
league_to_df = {}

def load_league_data():
    for league, csv_path in league_files.items():
        df = pd.read_csv(csv_path)
        # Beklenen kolonlar: HomeTeam, AwayTeam, FTHG, FTAG (Date opsiyonel)
        
        # Takım listesi
        teams = sorted(set(df['HomeTeam']).union(set(df['AwayTeam'])))
        
        league_to_teams[league] = teams
        league_to_df[league] = df

# Uygulama başlarken bir kez çağır
load_league_data()

#############################################################################
# Dixon-Coles yardımcı fonksiyonları
#############################################################################
def dixon_coles_correction_factor(home_goals, away_goals, rho):
    if home_goals == 0 and away_goals == 0:
        return 1 + rho
    elif (home_goals == 0 and away_goals == 1) or (home_goals == 1 and away_goals == 0):
        return 1 + rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1

def dixon_coles_log_likelihood(params, home_teams, away_teams, home_goals, away_goals, team_idx, rho):
    n_teams = len(team_idx)
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    
    log_likelihood = 0
    for i in range(len(home_goals)):
        hg = home_goals[i]
        ag = away_goals[i]
        
        h_idx = team_idx[home_teams[i]]
        a_idx = team_idx[away_teams[i]]
        
        lambda_home = math.exp(attack[h_idx] + defense[a_idx] + home_adv)
        lambda_away = math.exp(attack[a_idx] + defense[h_idx])
        
        match_log_prob = (
            -lambda_home + hg * math.log(lambda_home + 1e-10)
            -lambda_away + ag * math.log(lambda_away + 1e-10)
        )
        corr_factor = dixon_coles_correction_factor(hg, ag, rho)
        match_log_prob += math.log(corr_factor + 1e-10)
        
        log_likelihood += match_log_prob
    return -log_likelihood

def optimize_model(df, rho):
    teams = sorted(set(df['HomeTeam']).union(set(df['AwayTeam'])))
    team_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)
    
    initial_values = np.concatenate([
        np.random.uniform(-0.1, 0.1, n_teams),
        np.random.uniform(-0.1, 0.1, n_teams),
        [0.1]
    ])
    from scipy.optimize import minimize
    
    result = minimize(
        dixon_coles_log_likelihood,
        initial_values,
        args=(df['HomeTeam'], df['AwayTeam'], df['FTHG'], df['FTAG'], team_idx, rho),
        method='L-BFGS-B'
    )
    return result.x, teams, team_idx

#############################################################################
# Ana sayfa -> Form
#############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Sadece formu gösteriyoruz.
        # league_to_teams sözlüğünü JavaScript ile doldurmak için "league_data" olarak geçelim.
        return render_template("index.html", league_data=league_to_teams)
    
    # request.method == "POST"
    league = request.form.get("league")
    home_team = request.form.get("home_team")
    away_team = request.form.get("away_team")
    rho_str = request.form.get("rho", "-0.2")
    try:
        rho = float(rho_str)
    except ValueError:
        rho = -0.2
    
    # CSV verisini al
    df = league_to_df.get(league)
    if df is None:
        return "Seçili lig bulunamadı!", 400
    
    # Model eğit
    params_opt, teams, team_idx = optimize_model(df, rho)
    if home_team not in teams or away_team not in teams:
        return f"{home_team} veya {away_team} bu ligde bulunamadı.", 400
    
    n_teams = len(teams)
    attack_params = params_opt[:n_teams]
    defense_params = params_opt[n_teams:2*n_teams]
    home_adv = params_opt[2*n_teams]
    
    lambda_home = math.exp(attack_params[team_idx[home_team]] 
                           + defense_params[team_idx[away_team]] 
                           + home_adv)
    lambda_away = math.exp(attack_params[team_idx[away_team]] 
                           + defense_params[team_idx[home_team]])
    
    max_goals = 5
    prob_matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            poi_ij = (math.exp(-lambda_home) * (lambda_home**i) / math.factorial(i)) * \
                     (math.exp(-lambda_away) * (lambda_away**j) / math.factorial(j))
            dc_factor = dixon_coles_correction_factor(i, j, rho)
            prob_matrix[i,j] = poi_ij * dc_factor
    
    total_prob = prob_matrix.sum()
    if total_prob > 0:
        prob_matrix /= total_prob
    
    home_win_prob = np.sum(np.tril(prob_matrix, -1))
    draw_prob = np.sum(np.diag(prob_matrix))
    away_win_prob = np.sum(np.triu(prob_matrix, 1))
    
    # Geçmiş maç var mı?
    df_past_matches = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team))
        | ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ]
    
    if len(df_past_matches) == 0:
        past_text = "İki takım daha önce karşılaşmamış."
    else:
        last_match = df_past_matches.iloc[-1]
        dt = last_match.get('Date', '?')
        ht = last_match['HomeTeam']
        at = last_match['AwayTeam']
        hg = last_match['FTHG']
        ag = last_match['FTAG']
        past_text = f"Son maç {dt} tarihinde {ht} - {at} {hg}-{ag} bitti."
    
    result_text = (
        f"{league} liginde {home_team} - {away_team}\n"
        f"Ev Kazanma: %{100*home_win_prob:.2f}\n"
        f"Beraberlik: %{100*draw_prob:.2f}\n"
        f"Deplasman Kazanma: %{100*away_win_prob:.2f}\n"
        f"{past_text}"
    )
    
    # Isı haritası
    fig = plt.figure(figsize=(6, 5))
    plt.imshow(prob_matrix, origin='lower', aspect='auto', cmap='Blues')
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            plt.text(j, i, f"{prob_matrix[i,j]*100:.1f}%", ha="center", va="center", fontsize=8)
    plt.colorbar()
    plt.xticks(range(max_goals+1))
    plt.yticks(range(max_goals+1))
    plt.xlabel(f"{away_team} Golleri")
    plt.ylabel(f"{home_team} Golleri")
    plt.title(f"{home_team} - {away_team} (ρ={rho:.2f})")
    
    # Base64 dönüştür
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    # Tekrar index.html'e dönelim, sonucu gösterelim
    return render_template("index.html", league_data=league_to_teams, 
                           result_text=result_text, plot_url=plot_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
