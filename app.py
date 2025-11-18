import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
from scipy.stats import genextreme, genpareto, anderson, kstest, skew, kurtosis, norm
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="IDXQuality30 Portfolio Optimizer & Risk Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; padding-bottom: 10px;}
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding: 10px 0;}
    .info-text {background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin: 10px 0;}
    .result-box {background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #1f77b4;}
    .positive {color: #2e8b57; font-weight: bold;}
    .negative {color: #dc143c; font-weight: bold;}
    .data-frame {font-size: 0.9rem; margin-bottom: 20px;}
    .var-result {background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 5px solid #ffc107;}
    .welcome-container {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; color: white; margin-bottom: 30px;}
    .feature-card {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 10px 0;}
    .nav-button {background-color: #1f77b4; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px;}
    .nav-button:hover {background-color: #155a8a;}
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigasi Aplikasi")
page = st.sidebar.radio("Pilih Halaman:", 
                        ["🏠 Pengenalan Aplikasi", "📚 Tutorial & Panduan", "📊 Optimasi Portofolio", "🛡️ Analisis Risiko VaR"])

# Halaman 1: Pengenalan Aplikasi
if page == "🏠 Pengenalan Aplikasi":
    st.markdown("""
    <div class="welcome-container">
        <h1 style="color: white; text-align: center; margin-bottom: 20px;">🛡️ IDXQuality30 Portfolio Optimizer & Extreme Value Theory Analysis</h1>
        <h3 style="color: white; text-align: center; font-weight: 300;">
        Aplikasi ini mengoptimalkan portofolio saham IDXQuality30 dan melakukan analisis nilai ekstrem untuk menghitung Value at Risk menggunakan distribusi GEV dan GPD    </h3>
    </div>
    """, unsafe_allow_html=True)

    # Introduction Section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📊 Tentang Aplikasi Ini
        
        **IDXQuality30 Portfolio Optimizer & Extreme Value Theory Analysis** adalah aplikasi yang dirancang untuk membantu investor dalam:
        
        - **Mengoptimalkan portofolio** saham IDXQuality30 dengan pendekatan Markowitz
        - **Menganalisis risiko ekstrem** menggunakan metode Extreme Value Theory diantaranya distribusi Generalized Extreme Value (GEV) dan Generalized Pareto Distribution (GPD)
        - **Membantu pengambilan keputusan investasi** yang lebih mudah dipahami dan terukur
        """)

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135679.png", width=150)

    # Features Section
    st.markdown("""
    ### 🚀 Fitur Utama Aplikasi

    <div class="feature-card">
        <h4>🎯 Optimasi Portofolio Otomatis</h4>
        <ul>
        <li>Seleksi saham berbasis expected return positif dan perbandingan dengan risk free rate</li>
        <li>Pencarian target return optimal dengan nilai Sharpe ratio tertinggi</li>
        <li>Pembobotan portofolio yang efisien</li>
        <li>Visualisasi efficient frontier</li>
        </ul>
    </div>

    <div class="feature-card">
        <h4>📈 Analisis Nilai Ekstrem</h4>
        <ul>
        <li>Generalized Extreme Value (GEV) untuk block maxima</li>
        <li>Generalized Pareto Distribution (GPD) untuk tail distribution</li>
        <li>Goodness-of-fit testing untuk memastikan kecocokan distribusi</li>
        </ul>
    </div>

    <div class="feature-card">
        <h4>🛡️ Value at Risk Calculation</h4>
        <ul>
        <li>Perhitungan VaR harian dengan confidence level yang dapat disesuaikan oleh pengguna</li>
        <li>Perbandingan VaR menggunakan metode GEV dan GPD</li>
        <li>Rekomendasi distribusi terbaik berdasarkan AIC</li>
        <li>Konversi perhitungan VaR ke nilai Rupiah</li>
        </ul>
    </div>

    """, unsafe_allow_html=True)

    # Target Users Section
    st.markdown("""
    ### 👥 Untuk Siapa Aplikasi Ini?

    | Profil Investor | Manfaat yang Didapat |
    |----------------|---------------------|
    | **Investor Pemula** | Pemahaman dasar optimasi portofolio dan manajemen risiko |
    | **Investor Menengah** | Tools analisis canggih untuk meningkatkan decision making |
    | **Profesional Keuangan** | Metodologi EVT untuk risk management yang lebih akurat |
    | **Akademisi/Peneliti** | Implementasi praktis teori portofolio dan extreme value theory analysis |
    """)

# Halaman 2: Tutorial & Panduan
elif page == "📚 Tutorial & Panduan":
    st.markdown('<div class="section-header">📚 Panduan Penggunaan Aplikasi</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Panduan Penggunaan

    #### Langkah 1: Optimasi Portofolio
    - Pilih rentang tanggal data historis pada side bar
    - Masukkan risk-free rate yang diperoleh dari rata-rata BI7DayRate yang diunduh pada link ini https://www.bi.go.id/id/statistik/indikator/bi-rate.aspx
    - Tentukan periode yang disamakan dengan periode yang dianalisis
    - Masukkan jumlah investasi dalam Rupiah
    - Pilih saham yang akan dianalisis (Dapat memilih minimal 3 dengan menghilangkan centang pada kotak "pilih semua")
    - Setelah dipastikan ulang dan sesuai maka klik tombol "Optimasi portofolio" untuk melakukan analisis

    #### Langkah 2: Analisis Risiko VaR
    - Atur ukuran blok untuk GEV (5-30 hari) jika periode yang diinginkan mingguan maka 5 hari, dan berlaku kelipatan yang dapat diisi bebas oleh pengguna
    - Pilih tingkat keyakinan VaR (90-99%)
    - Otomatis akan menghitung VaR yang disesuaikan dengan portofolio optimal pada optimasi portofolio
                

    ### 💡 Tips Penggunaan Portofolio Optimal

    **Best Practices**:
    - Gunakan data historis minimal 3 tahun
    - Update parameter secara berkala

    ### 🆘 Bantuan dan Support

    Jika Anda mengalami kesulitan atau memiliki pertanyaan:
    1. Periksa rentang tanggal data yang digunakan
    2. Pastikan koneksi internet stabil untuk download data
    3. Sesuaikan parameter analisis risiko VaR sesuai kebutuhan

    ---

    **Selamat Menggunakan Aplikasi! 🎉**

    *Mulai dengan memilih menu di sidebar yang terdiri dari Optimalisasi Portfolio & Analisis Risiko VaR (Proses dimulai dari Optimasi Portofolio dan dilanjutkan analisis risiko VaR)"*
    """)

# Halaman 3: Optimasi Portofolio
elif page == "📊 Optimasi Portofolio":
    st.markdown('<div class="section-header">📊 Optimasi Portofolio IDXQuality30</div>', unsafe_allow_html=True)

    # List saham Indeks IDXQuality30
    idx_quality30 = [
        'ACES.JK', 'ADRO.JK', 'AKRA.JK', 'AMRT.JK', 'ASII.JK',
        'BBCA.JK', 'BBRI.JK', 'BFIN.JK', 'BMRI.JK', 'BNGA.JK',
        'BRIS.JK', 'BTPS.JK', 'CPIN.JK', 'INCO.JK', 'INTP.JK',
        'KLBF.JK', 'LSIP.JK', 'MIKA.JK', 'MNCN.JK', 'MYOR.JK',
        'NISP.JK', 'PTBA.JK', 'SCMA.JK', 'SIDO.JK', 'TLKM.JK',
        'UNTR.JK'
    ]


    # Sidebar untuk input pengguna 
    with st.sidebar:
        st.header("Parameter Investasi")
        
        # Seleksi tanggal
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365*3)  # Default to 3 years of data
        
        col1, col2 = st.columns(2)
        with col1:
            start_input = st.date_input("Start Date", value=start_date)
        with col2:
            end_input = st.date_input("End Date", value=end_date)
        
        # Risk-free rate input
        risk_free_annual = st.number_input(
            "Annual Risk-Free Rate (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=4.94, 
            step=0.1,
            help="Default is 4.94% annually"
        )
        
        # Convert annual risk-free rate ke harian
        risk_free_daily = (1 + risk_free_annual/100) ** (1/252) - 1
        
        #Pemilihan Target Return Optimal Otomatis
        st.info("Target return akan dipilih otomatis untuk Sharpe ratio tertinggi")
        
        # Jumlah investasi
        investment_amount = st.number_input(
            "Jumlah Investasi (IDR)", 
            min_value=100000, 
            value=100000000, 
            step=1000000,
            help="Masukkan jumlah yang ingin anda investasikan dalam Rupiah Indonesia"
        )
        
        # Pemilihan saham
        st.subheader("Pemilihan saham")
        select_all = st.checkbox("Pilih semua saham", value=True)
        
        if select_all:
            selected_stocks = idx_quality30
        else:
            selected_stocks = st.multiselect(
                "Pilih saham yang akan dimasukkan:",
                idx_quality30,
                default=idx_quality30[:5]  # Default to first 5 stocks
            )
        
        # Calculate button
        calculate_button = st.button("Optimalisasi Portfolio", type="primary")

    # Main content area untuk optimasi portofolio
    if not selected_stocks:
        st.warning("Silakan pilih setidaknya satu saham untuk dianalisis.")
        st.stop()

    if calculate_button or 'portfolio_data' in st.session_state:
        if calculate_button:
            # Clear session state jika menghitung ulang
            if 'portfolio_data' in st.session_state:
                del st.session_state['portfolio_data']
            if 'show_risk_analysis' in st.session_state:
                del st.session_state['show_risk_analysis']
                
        with st.spinner("Mengambil data dan mengoptimalkan portofolio..."):
            try:
                # Download data saham
                stock_data = yf.download(selected_stocks, start=start_input, end=end_input)['Close']
                
                if stock_data.empty:
                    st.error("Tidak ada data yang tersedia untuk rentang tanggal yang dipilih. Silakan coba rentang tanggal yang berbeda.")
                    st.stop()
                
                # Urutkan data dari yang terlama hingga yang terbaru
                stock_data = stock_data.sort_index(ascending=True)
                
                # Tampilkan data saham asli
                st.markdown('<div class="section-header">Data Harga Saham </div>', unsafe_allow_html=True)
                st.dataframe(stock_data, use_container_width=True)
                
                # Missing value handling 
                st.markdown('<div class="section-header">Interpolasi Linear untuk Missing Values</div>', unsafe_allow_html=True)
                
                # Fungsi untuk interpolasi linear manual sesuai rumus: nilai_sebelumnya + ((nilai_setelahnya - nilai_sebelumnya)/2)*1
                def manual_linear_interpolation(series):
                    interpolated_series = series.copy()
                    
                    for i in range(len(interpolated_series)):
                        if pd.isna(interpolated_series.iloc[i]):
                            # Cari nilai sebelumnya yang tidak missing value
                            prev_idx = i - 1
                            while prev_idx >= 0 and pd.isna(interpolated_series.iloc[prev_idx]):
                                prev_idx -= 1
                            
                            # Cari nilai setelahnya yang tidak missing value
                            next_idx = i + 1
                            while next_idx < len(interpolated_series) and pd.isna(interpolated_series.iloc[next_idx]):
                                next_idx += 1
                            
                            # Jika ditemukan nilai sebelum dan setelahnya
                            if prev_idx >= 0 and next_idx < len(interpolated_series):
                                prev_val = interpolated_series.iloc[prev_idx]
                                next_val = interpolated_series.iloc[next_idx]
                                
                                # Rumus interpolasi linear: nilai_sebelumnya + ((nilai_setelahnya - nilai_sebelumnya)/2)*1
                                interpolated_value = prev_val + ((next_val - prev_val) / 2) * 1
                                interpolated_series.iloc[i] = interpolated_value
                    
                    return interpolated_series

                stock_data_interpolated = stock_data.copy()
                for column in stock_data_interpolated.columns:
                    stock_data_interpolated[column] = manual_linear_interpolation(stock_data_interpolated[column])
                
                st.dataframe(stock_data_interpolated, use_container_width=True)
                
                # Menghitung expected return harian dari data yang diinterpolasi
                returns = stock_data_interpolated.pct_change().dropna()
                
                st.markdown('<div class="section-header">Perhitungan Returns Harian</div>', unsafe_allow_html=True)
                st.dataframe(returns.tail().style.format("{:.4%}"), use_container_width=True)
                
                # Menghitung expected return menggunakan geometric mean (harian).

                # Fungsi untuk menghitung geometric mean
                def geometric_mean(series):
                    return (np.prod(1 + series) ** (1 / len(series)) - 1)

                geometric_returns = returns.apply(geometric_mean)
                geometric_returns_original_order = geometric_returns.copy()
                
                st.markdown('<div class="section-header">Expected Returns Harian (Geometric Mean)</div>', unsafe_allow_html=True)
                expected_returns_df = pd.DataFrame({
                    'Stock': geometric_returns_original_order.index,
                    'Expected Return Harian': geometric_returns_original_order.values
                })
                st.dataframe(expected_returns_df.style.format({'Expected Return Harian': '{:.4%}'}), use_container_width=True)
                
                # Hapus saham yang memiliki expected return negatif.
                st.markdown('<div class="section-header">Seleksi Saham (Expected Return Positif)</div>', unsafe_allow_html=True)
                positive_return_stocks = geometric_returns_original_order[geometric_returns_original_order > 0]
                st.write(f"Saham dengan expected return positif: {len(positive_return_stocks)} of {len(geometric_returns_original_order)}")
                
                # Hapus saham yang memiliki expected return kurang dari risk-free rate
                st.markdown('<div class="section-header">Seleksi Saham (Expected Return > Risk-Free Rate)</div>', unsafe_allow_html=True)
                qualified_stocks = positive_return_stocks[positive_return_stocks > risk_free_daily]
                st.write(f"Saham dengan expected returns > risk-free rate harian: {len(qualified_stocks)} of {len(positive_return_stocks)}")
                
                # Tampilkan saham yang telah difilter
                if len(qualified_stocks) > 0:
                    st.write("Saham yang memenuhi syarat setelah seleksi:")
                    qualified_df = pd.DataFrame({
                        'Saham': qualified_stocks.index,
                        'Expected Return Harian': qualified_stocks.values,
                        'Risk-Free Rate Harian': risk_free_daily
                    })
                    st.dataframe(qualified_df.style.format({
                        'Expected Return Harian': '{:.4%}', 
                        'Risk-Free Rate Harian': '{:.4%}'
                    }), use_container_width=True)
                    
                    # Use only qualified stocks for portfolio optimization
                    selected_stocks_filtered = qualified_stocks.index.tolist()
                    
                    # Pastikan returns yang digunakan adalah dari data yang sudah diinterpolasi
                    returns_filtered = returns[selected_stocks_filtered]
                    
                    # Recalculate expected returns for filtered stocks 
                    expected_returns_filtered = qualified_stocks

                    # Fungsi untuk menghitung weighted variance-covariance matrix
                    def calculate_weighted_covariance_matrix(weights, cov_matrix):
                        n = len(weights)
                        weighted_cov_matrix = np.zeros((n, n))
                        
                        for i in range(n):
                            for j in range(n):
                                weighted_cov_matrix[i, j] = weights[i] * weights[j] * cov_matrix.iloc[i, j]
                        
                        return weighted_cov_matrix
                    
                    # Calculate covariance matrix for filtered stocks (daily) 
                    cov_matrix = returns_filtered.cov()
                    
                    # Konversi ke numpy array untuk konsistensi dengan Excel
                    cov_matrix_np = cov_matrix.to_numpy()
                    
                    # Heatmap korelasi
                    st.markdown('<div class="section-header">Heatmap Korelasi Saham</div>', unsafe_allow_html=True)
                    
                    # Hitung matriks korelasi
                    correlation_matrix = returns_filtered.corr()
                    
                    # Tampilkan heatmap korelasi
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(correlation_matrix, 
                               annot=True, 
                               cmap='coolwarm', 
                               center=0,
                               square=True,
                               fmt='.3f',
                               cbar_kws={'shrink': 0.8})
                    plt.title('Heatmap Korelasi Antar Saham', fontsize=16, fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    st.pyplot(fig)
                                    
                    # Target expected return options from the image
                    target_return_options = [
                        0.00025, 0.00030, 0.00035, 0.00040, 0.00045, 0.00050, 0.00055, 0.00060,
                        0.00065, 0.00070, 0.00075, 0.00080, 0.00085, 0.00090, 0.00095, 0.00100,
                        0.00105, 0.00110
                    ]                    
                    
                    # Fungsi untuk menghitung statistik portofolio (harian)
                    def portfolio_stats(weights):
                        # Expected return portofolio
                        port_return = np.dot(weights, expected_returns_filtered)
                        
                        # Varians Portofolio
                        port_variance = np.dot(weights.T, np.dot(cov_matrix_np, weights))
                        
                        # Standar deviasi
                        port_volatility = np.sqrt(port_variance)
                        
                        # Sharpe ratio
                        sharpe_ratio = (port_return - risk_free_daily) / port_volatility if port_volatility > 0 else 0
                        
                        return port_return, port_volatility, sharpe_ratio, port_variance
                    
                    # Tentukan bobot portofolio optimal untuk tingkat pengembalian target 
                    # Tentukan fungsi tujuan (meminimalkan varians/volatilitas)
                    def portfolio_variance(weights):
                        port_return, port_volatility, sharpe_ratio , port_variance = portfolio_stats(weights)
                        return port_variance
                    
                    # Batasan: bobot antara 0 dan 1 (tidak ada penjualan pendek)
                    bounds = tuple((0.0, 1.0) for _ in range(len(selected_stocks_filtered)))
                    
                    # Tebakan awal (bobot sama)
                    init_guess = np.array([1.0/len(selected_stocks_filtered)] * len(selected_stocks_filtered))
                    
                    
                    # OTOMATIS MEMILIH TARGET RETURN DENGAN SHARPE RATIO TERTINGGI
                    
                    st.markdown('<div class="section-header">Pembobotan Portofolio</div>', unsafe_allow_html=True)

                    # Cari target return yang memberikan Sharpe ratio tertinggi
                    best_sharpe = -np.inf
                    best_target_return = None
                    best_optimal_weights = None
                    best_portfolio_stats = None
                    optimization_results = []

                    for target_ret in target_return_options:
                        try:
                            # Batasan untuk target return
                            constraints = [
                                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
                                {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns_filtered) - target_ret}  # target return
                            ]
                            
                            # Optimalkan untuk varians minimum dengan target pengembalian yang diberikan.
                            opt_results = minimize(
                                portfolio_variance, 
                                init_guess, 
                                method='SLSQP', 
                                bounds=bounds, 
                                constraints=constraints,
                                options={'ftol': 1e-10, 'disp': False}
                            )
                            
                            if opt_results.success:
                                weights = opt_results.x
                                # Normalisasi bobot untuk memastikan totalnya sama dengan 1.
                                weights = weights / np.sum(weights)
                                port_return, port_volatility, sharpe_ratio, port_variance = portfolio_stats(weights)
                                
                                # Simpan weights untuk setiap saham
                                weight_dict = {}
                                for i, stock in enumerate(selected_stocks_filtered):
                                    weight_dict[stock] = weights[i]
                                
                                optimization_results.append({
                                    'Target Return': target_ret,
                                    'Actual Return': port_return,
                                    'Volatility': port_volatility,
                                    'Sharpe Ratio': sharpe_ratio,
                                    'Variance': port_variance,
                                    'Weights': weight_dict,
                                    'Success': True
                                })
                                
                                # Perbarui portofolio terbaik jika Sharpe ratio lebih tinggi.
                                if sharpe_ratio > best_sharpe:
                                    best_sharpe = sharpe_ratio
                                    best_target_return = target_ret
                                    best_optimal_weights = weights
                                    best_portfolio_stats = (port_return, port_volatility, sharpe_ratio, port_variance)
                        
                        except Exception as e:
                            optimization_results.append({
                                'Target Return': target_ret,
                                'Actual Return': np.nan,
                                'Volatility': np.nan,
                                'Sharpe Ratio': np.nan,
                                'Variance': np.nan,
                                'Weights': {},
                                'Success': False
                            })
                            continue

                    # Tampilkan hasil pencarian target return optimal dengan bobot
                    st.write("Hasil optimasi untuk berbagai target return:")

                    # Membuat DataFrame yang menampilkan weights untuk setiap target return
                    optimization_display_data = []
                    for result in optimization_results:
                        if result['Success']:
                            row_data = {
                                'Target Return': result['Target Return'] * 100,  # Convert to percentage
                                'Sharpe Ratio': result['Sharpe Ratio']*100, #Convert to Percentage
                                'Volatility': result['Volatility'] * 100  # Convert to percentage
                            }
                            # Tambahkan weights untuk setiap saham
                            for stock, weight in result['Weights'].items():
                                row_data[stock] = weight * 100  # Convert to percentage
                            optimization_display_data.append(row_data)

                    if optimization_display_data:
                        optimization_df = pd.DataFrame(optimization_display_data)
                        # Format untuk display - semua dalam persentase
                        format_dict = {
                            'Target Return': '{:.3f}%', 
                            'Sharpe Ratio': '{:.4f}%', 
                            'Volatility': '{:.3f}%'
                        }
                        for stock in selected_stocks_filtered:
                            if stock in optimization_df.columns:
                                format_dict[stock] = '{:.2f}%'
                        
                        st.dataframe(optimization_df.style.format(format_dict), use_container_width=True)

                    if best_target_return is not None:
                        optimal_return, optimal_volatility, optimal_sharpe, optimal_variance = best_portfolio_stats
                        optimal_weights = best_optimal_weights           

                        # Efficient Frontier                        
                        # Cari portfolio dengan standar deviasi terkecil
                        min_volatility = np.inf
                        min_vol_weights = None
                        min_vol_stats = None
                        
                        # Mengumpulkan data untuk efficient frontier
                        efficient_portfolios = []
                        
                        for target_ret in target_return_options:
                            try:
                                constraints = [
                                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                                    {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns_filtered) - target_ret}
                                ]
                                
                                opt_results = minimize(
                                    portfolio_variance, 
                                    init_guess, 
                                    method='SLSQP', 
                                    bounds=bounds, 
                                    constraints=constraints,
                                    options={'ftol': 1e-10, 'disp': False}
                                )
                                
                                if opt_results.success:
                                    weights = opt_results.x
                                    weights = weights / np.sum(weights)
                                    port_return, port_volatility, sharpe_ratio, port_variance = portfolio_stats(weights)
                                    
                                    efficient_portfolios.append({
                                        'Target Return (%)': target_ret * 100,
                                        'Actual Return (%)': port_return * 100,
                                        'Volatility (%)': port_volatility * 100,
                                        'Variance': port_variance,
                                        'Sharpe Ratio': sharpe_ratio
                                    })
                                    
                                    if port_volatility < min_volatility:
                                        min_volatility = port_volatility
                                        min_vol_weights = weights
                                        min_vol_stats = (port_return, port_volatility, sharpe_ratio, port_variance)
                            
                            except Exception as e:
                                continue
                        
                        if min_vol_weights is not None:
                            efficient_return, efficient_volatility, efficient_sharpe, efficient_variance = min_vol_stats
                            
                        # Visualisasi Efficient Frontier
                        st.subheader("Visualisasi Efficient Frontier")

                        # Membuat dataframe efficient frontier
                        efficient_df = pd.DataFrame(efficient_portfolios)

                        # Membuat visualisasi Efficient Frontier
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(efficient_df['Volatility (%)'], efficient_df['Actual Return (%)'], 
                                            c=efficient_df['Sharpe Ratio'], cmap='viridis', marker='o', s=50)
                        # Menandai portfolio optimal
                        ax.scatter(optimal_volatility * 100, optimal_return * 100, c='red', marker='*', s=200, 
                                label=f'Portfolio Optimal (Sharpe: {optimal_sharpe:.4f})')

                        # Menandai portfolio efisien
                        if min_vol_weights is not None:
                            ax.scatter(efficient_volatility * 100, efficient_return * 100, c='blue', marker='s', s=150, 
                                    label=f'Portfolio Efisien (Vol: {efficient_volatility*100:.4f}%)')

                        ax.set_xlabel('Volatility (%)')
                        ax.set_ylabel('Return (%)')
                        ax.set_title('Efficient Frontier\n')
                        ax.legend()
                        ax.grid(True)
                        plt.colorbar(scatter, label='Sharpe Ratio')
                        st.pyplot(fig)

                        # PEMBOBOTAN PORTOFOLIO EFISIEN (STANDAR DEVIASI TERKECIL)
                        
                        st.markdown('<div class="section-header">Portfolio Efisien - Portofolio dengan risiko terkecil</div>', unsafe_allow_html=True)

                        st.info( 
                            "Portofolio efisien dengan risiko terendah ditujukan untuk investor yang **sangat menghindari risiko.** "
                        )

                        if min_vol_weights is not None:
                            efficient_return, efficient_volatility, efficient_sharpe, efficient_variance = min_vol_stats
                            
                            # Menampilkan statistik portofolio yang efisien

                            st.subheader("Statistik Portfolio Efisien")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.markdown(f'<div class="result-box">Expected Return Harian: <span class="positive">{efficient_return*100:.4f}%</span></div>', unsafe_allow_html=True)
                            with col2:
                                st.markdown(f'<div class="result-box">Expected Volatility Harian: <span class="negative">{efficient_volatility*100:.4f}%</span></div>', unsafe_allow_html=True)
                            with col3:
                                st.markdown(f'<div class="result-box">Sharpe Ratio Harian: <span class="positive">{efficient_sharpe:.4f}</span></div>', unsafe_allow_html=True)
                            with col4:
                                st.markdown(f'<div class="result-box">Variance Portofolio: <span class="negative">{efficient_variance:.8f}</span></div>', unsafe_allow_html=True)
                            
                            # Menampilkan pembobotan portofolio efisien 
                            st.markdown(f'<div class="section-header">Proporsi Bobot Saham Efisien</div>', unsafe_allow_html=True)
                            
                            # Buat DataFrame untuk bobot portofolio efisien
                            efficient_weights_data = []
                            for i, stock in enumerate(selected_stocks_filtered):
                                efficient_weights_data.append({
                                    'Stock': stock,
                                    'Weight Asli': min_vol_weights[i],
                                    'Weight Asli (%)': min_vol_weights[i] * 100
                                })

                            efficient_weights_df = pd.DataFrame(efficient_weights_data)

                            # Hitung bobot bulat untuk portofolio efisien
                            efficient_weights_df['Weight Bulat (%)'] = np.round(efficient_weights_df['Weight Asli (%)'])

                            # Normalisasi bobot bulat agar totalnya 100%
                            total_rounded_efficient = efficient_weights_df['Weight Bulat (%)'].sum()
                            if total_rounded_efficient != 100:
                                # Menyesuaikan bobot terbesar agar totalnya menjadi 100%
                                diff_efficient = 100 - total_rounded_efficient
                                max_idx_efficient = efficient_weights_df['Weight Bulat (%)'].idxmax()
                                efficient_weights_df.loc[max_idx_efficient, 'Weight Bulat (%)'] += diff_efficient

                            # Hitung alokasi dana berdasarkan bobot bulat
                            efficient_weights_df['Amount (IDR)'] = (efficient_weights_df['Weight Bulat (%)'] / 100 * investment_amount).round(2)

                            # Tampilkan dalam urutan asli
                            st.dataframe(efficient_weights_df[['Stock', 'Weight Bulat (%)', 'Amount (IDR)']], 
                                        use_container_width=True, hide_index=True)

                            # Verifikasi total bobot
                            st.write(f"**Total Bobot Portfolio Efisien: {efficient_weights_df['Weight Bulat (%)'].sum()}%**")

                        # OTOMATIS MENGGUNAKAN PORTFOLIO OPTIMAL UNTUK ANALISIS SELANJUTNYA
                  
                        selected_weights = optimal_weights
                        selected_return = optimal_return
                        selected_volatility = optimal_volatility
                        selected_sharpe = optimal_sharpe
                        selected_variance = optimal_variance
                        portfolio_type = "Optimal"
                        
                        # Tampilkan kinerja portofolio dan risikonya
                        st.markdown(f'<div class="section-header">Portofolio {portfolio_type} - Portofolio dengan Indeks Sharpe Tertinggi</div>', unsafe_allow_html=True)
                        
                        st.info( 
                            "Portofolio optimal dengan indeks sharpe terbesar ditujukan untuk investor yang **fokus memaksimalkan return yang disesuaikan dengan risiko seimbang.** "
                        )

                        st.subheader("Statistik Portfolio Optimal")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f'<div class="result-box">Expected Return Harian: <span class="positive">{selected_return*100:.4f}%</span></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="result-box">Expected Volatility Harian: <span class="negative">{selected_volatility*100:.4f}%</span></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="result-box">Sharpe Ratio Harian: <span class="positive">{selected_sharpe:.4f}</span></div>', unsafe_allow_html=True)
                        with col4:
                            st.markdown(f'<div class="result-box">Variance Portofolio: <span class="negative">{selected_variance:.8f}</span></div>', unsafe_allow_html=True)
                        
                        # Menampilkan bobot terpilih dalam bentuk bilangan bulat
                        st.markdown(f'<div class="section-header">Proporsi Bobot Saham {portfolio_type}</div>', unsafe_allow_html=True)

                        # Buat DataFrame langsung dari bobot terpilih
                        weights_data = []
                        for i, stock in enumerate(selected_stocks_filtered):
                            weights_data.append({
                                'Stock': stock,
                                'Weight Asli': selected_weights[i],
                                'Weight Asli (%)': selected_weights[i] * 100
                            })

                        weights_df = pd.DataFrame(weights_data)

                        # Hitung bobot bulat
                        weights_df['Weight Bulat (%)'] = np.round(weights_df['Weight Asli (%)'])

                        # Normalisasi bobot bulat agar totalnya 100%
                        total_rounded = weights_df['Weight Bulat (%)'].sum()
                        if total_rounded != 100:
                            # Menyesuaikan bobot terbesar agar totalnya menjadi 100%
                            diff = 100 - total_rounded
                            max_idx = weights_df['Weight Bulat (%)'].idxmax()
                            weights_df.loc[max_idx, 'Weight Bulat (%)'] += diff

                        # Hitung alokasi dana berdasarkan bobot bulat
                        weights_df['Amount (IDR)'] = (weights_df['Weight Bulat (%)'] / 100 * investment_amount).round(2)

                        # Tampilkan dalam urutan asli (tidak diurutkan berdasarkan bobot)
                        st.dataframe(weights_df[['Stock', 'Weight Bulat (%)', 'Amount (IDR)']], 
                                    use_container_width=True, hide_index=True)

                        # Verifikasi total bobot
                        st.write(f"**Total Bobot: {weights_df['Weight Bulat (%)'].sum()}%**")
                        
                        # Menghitung portfolio returns harian berdasarkan bobot yang dipilih.
                        portfolio_daily_returns = (returns_filtered * selected_weights).sum(axis=1)
                        
                        # Kinerja portofolio dari waktu ke waktu 
                        st.markdown('<div class="section-header">Kinerja portofolio dari waktu ke waktu</div>', unsafe_allow_html=True)

                        # Menghitung return kumulatif portofolio optimal
                        cumulative_returns = (1 + portfolio_daily_returns).cumprod()

                        # Konversi ke persentase 
                        cumulative_returns_percent = (cumulative_returns - 1) * 100

                        # Membuat DataFrame untuk return kumulatif dalam persentase dengan format string
                        cumulative_df = pd.DataFrame({
                            'Date': cumulative_returns.index,
                            'Cumulative_Return_Percent': cumulative_returns_percent.values
                        })

                        # Format kolom persentase untuk CSV (menambahkan %)
                        cumulative_df_download = cumulative_df.copy()
                        cumulative_df_download['Cumulative_Return_Percent'] = cumulative_df_download['Cumulative_Return_Percent'].apply(lambda x: f'{x:.3f}%')

                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.plot(cumulative_returns_percent.index, cumulative_returns_percent.values)
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Cumulative Return (%)')
                        ax.set_title('Historical Performance of Optimal Portfolio\n')
                        ax.grid(True)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)

                        # Download button untuk data return kumulatif dalam format persen
                        st.subheader("Download Data Cumulative Returns")
                        csv = cumulative_df_download.to_csv(index=False)
                        st.download_button(
                            label="Download Cumulative Returns as CSV",
                            data=csv,
                            file_name=f"cumulative_returns_{portfolio_type.lower()}.csv",
                            mime="text/csv"
                        )
                        
                        # Simpan data portofolio ke session state untuk digunakan di halaman risiko
                        st.session_state['portfolio_data'] = {
                            'selected_weights': selected_weights,
                            'selected_stocks_filtered': selected_stocks_filtered,
                            'returns_filtered': returns_filtered,
                            'portfolio_daily_returns': portfolio_daily_returns,
                            'investment_amount': investment_amount
                        }
                        
                    else:
                        st.error("Tidak dapat menemukan portofolio optimal. Silakan coba parameter yang berbeda.")
                    
                else:
                    st.warning("Tidak ada saham yang memenuhi kriteria seleksi. Silakan sesuaikan periode, annual risk free rate, jumlah investasi, atau pilih saham yang berbeda.")
                
            except Exception as e:
                st.error(f"An error occurred during calculation: {str(e)}")
                import traceback
                st.write("Detail error:", traceback.format_exc())

# Halaman 4: Analisis Risiko VaR
elif page == "🛡️ Analisis Risiko VaR":
    st.markdown('<div class="section-header">🛡️ Analisis Risiko Value at Risk (VaR)</div>', unsafe_allow_html=True)
    
    if 'portfolio_data' not in st.session_state:
        st.warning("Silakan lakukan optimasi portofolio terlebih dahulu di halaman 'Optimasi Portofolio'.")
        if st.button("Kembali ke Optimasi Portofolio"):
            st.session_state['show_risk_analysis'] = False
            st.experimental_rerun()
        st.stop()
    
    # Sidebar untuk parameter EVT
    with st.sidebar:
        st.header("Parameter Extreme Value Theory")
        
        block_size = st.slider(
            "Ukuran blok untuk GEV (Hari)",
            min_value=5,
            max_value=30,
            value=5,
            help="Ukuran blok untuk Block Maxima Method (GEV)"
        )
        
        gpd_threshold_percentile = 10
        st.write("Threshold Percentile untuk GPD (%) =", gpd_threshold_percentile)

        
        var_confidence = st.slider(
            "Tingkat keyakinan VaR (%)",
            min_value=90,
            max_value=99,
            value=95,
            help="Tingkat kepercayaan untuk Value at Risk"
        )
        
        # Tombol untuk kembali ke optimasi
        st.markdown("---")
        if st.button("⬅️ Kembali ke Optimasi Portofolio"):
            st.session_state['show_risk_analysis'] = False
            st.experimental_rerun()
    
    # Ambil data portofolio dari session state
    portfolio_data = st.session_state['portfolio_data']
    selected_weights = portfolio_data['selected_weights']
    selected_stocks_filtered = portfolio_data['selected_stocks_filtered']
    returns_filtered = portfolio_data['returns_filtered']
    portfolio_daily_returns = portfolio_data['portfolio_daily_returns']
    investment_amount = portfolio_data['investment_amount']
    
    # Konversi ke DataFrame untuk konsistensi
    return_porto = pd.DataFrame({'return': portfolio_daily_returns})
    returns_series = return_porto["return"]
    
    with st.spinner("Melakukan analisis risiko VaR..."):
        st.info( 
            " Analisis Value at Risk difokuskan pada portofolio optimal, sedangkan portofolio efisien hanya digunakan sebagai pembanding untuk investor"
        )
        # EXTREME VALUE THEORY SECTION
                                
        st.markdown('<div class="section-header">Analisis Statistik Deskriptif Return Portofolio</div>', unsafe_allow_html=True)
        
        # Menampilkan statistika deskriptif
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Return Harian", f"{returns_series.mean()*100:.4f}%")
        with col2:
            st.metric("Std Dev Harian", f"{returns_series.std()*100:.4f}%")
        with col3:
            st.metric("Min Return", f"{returns_series.min()*100:.4f}%")
        with col4:
            st.metric("Max Return", f"{returns_series.max()*100:.4f}%")
        
        # Menghitung skewness and kurtosis
        skewness_value = skew(returns_series)
        kurtosis_value = kurtosis(returns_series, fisher=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Skewness", f"{skewness_value:.4f}")
        with col2:
            st.metric("Kurtosis (Fisher)", f"{kurtosis_value:.4f}")
        
        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(returns_series, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(returns_series.mean(), color='red', linestyle='--', label=f'Mean: {returns_series.mean()*100:.4f}%')
            ax.set_xlabel('Return')
            ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi Return Portofolio Optimal')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(returns_series)
            ax.set_ylabel('Return')
            ax.set_title('Boxplot Return Portofolio Optimal')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Normality test
        st.markdown('<div class="section-header">Uji Normalitas</div>', unsafe_allow_html=True)
        
        # Menghitung parameters untuk distribusi normal
        mean = np.mean(returns_series)
        std = np.std(returns_series)
        
        # Kolmogorov-Smirnov test untuk normalitas
        ks_statistic, p_value_norm = kstest(returns_series, "norm", args=(mean, std))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Kolmogorov-Smirnov Statistic", f"{ks_statistic:.4f}")
        with col2:
            st.metric("P-value", f"{p_value_norm:.4f}")
        
        alpha = 0.05
        if p_value_norm > alpha:
            st.success("Data return portofolio berdistribusi normal pada alpha 5% (H0 diterima).")
        else:
            st.warning("Data return portofolio TIDAK berdistribusi normal pada alpha 5% (H0 ditolak).")
        
        # GEV ANALYSIS

        st.markdown('<div class="section-header">Generalized Extreme Value (GEV) Analysis</div>', unsafe_allow_html=True)
        
        # Block Maxima untuk GEV
        block_maxima = [min(returns_series[i:i+block_size]) for i in range(0, len(returns_series), block_size)]
        
        st.write(f"Jumlah blok minima (ukuran blok {block_size} hari): {len(block_maxima)}")
        
        # Distribusi GEV yang sesuai
        gev_params = genextreme.fit(block_maxima)
        shape_gev, loc_gev, scale_gev = gev_params
        
        # Menghitung statistik GEV
        log_likelihood_gev = np.sum(genextreme.logpdf(block_maxima, shape_gev, loc_gev, scale_gev))
        AIC_gev = 2 * 3 - 2 * log_likelihood_gev  # 3 parameters
        
        # Menampilkan parameter GEV
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shape Parameter (ξ)", f"{shape_gev:.6f}")
        with col2:
            st.metric("Location Parameter (μ)", f"{loc_gev:.6f}")
        with col3:
            st.metric("Scale Parameter (σ)", f"{scale_gev:.6f}")
        with col4:
            st.metric("AIC GEV", f"{AIC_gev:.4f}")
        
        # GEV goodness-of-fit test
        ks_stat_gev, p_value_gev = kstest(block_maxima, "genextreme", args=(shape_gev, loc_gev, scale_gev))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("KS Statistic GEV", f"{ks_stat_gev:.4f}")
        with col2:
            st.metric("P-value GEV", f"{p_value_gev:.4f}")
        
        if p_value_gev > alpha:
            st.success("Distribusi GEV adalah fit yang baik untuk data minima pada alpha 5% (H0 diterima).")
            gev_good_fit = True
        else:
            st.warning("Distribusi GEV mungkin bukan fit yang baik untuk data minima pada alpha 5% (H0 ditolak).")
            gev_good_fit = False
        
        # Visualisasi GEV
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot 50 Data Return Portofolio Pertama dengan skala 0-50
        returns_50 = returns_series.iloc[:50]  # Mengambil 50 data return portofolio pertama
        x_positions = np.arange(len(returns_50))  # Buat array [0, 1, 2, ..., 49]

        # Plot semua return dengan warna biru
        ax.plot(x_positions, returns_50.values, 'bo-', markersize=4, linewidth=1, alpha=0.7, label='Return Portofolio')

        # Menandai nilai terendah dengan titik merah + nilai angkanya dalam persen
        for i in range(0, len(returns_50), block_size):
            block_end = min(i + block_size, len(returns_50))
            block_data = returns_50.iloc[i:block_end]
            min_index = block_data.idxmin()
            min_value = block_data.min()
            
            # Mencari posisi x dari minima
            min_x_position = np.where(returns_50.index == min_index)[0][0]
            
            # Plot titik minima dengan warna merah
            ax.scatter(min_x_position, min_value, color='red', s=80, zorder=5)
            
            # Tambahkan label nilai dalam persen di dekat titik minima
            ax.text(min_x_position, min_value,
                    f'{min_value*100:.3f}%',  # tampilkan dalam persen
                    color='red', fontsize=9, fontweight='bold',
                    ha='left', va='bottom')

        ax.set_xlabel('Data Point')
        ax.set_ylabel('Return (%)')
        ax.set_title('50 Data Return Portofolio Pertama\n')
        ax.set_xticks([0, 10, 20, 30, 40, 49])  # Set ticks di 0,10,20,30,40,50
        ax.set_xticklabels(['0', '10', '20', '30', '40', '50'])
        ax.grid(True, alpha=0.3)

        # Menambahkan legend manual
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='blue', linestyle='-', markersize=6, label='Return Portofolio'),
            Line2D([0], [0], marker='o', color='red', linestyle='None', markersize=8, label='Return Terkecil (Nilai Ekstrem)')
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        st.pyplot(fig)
                        

        # GPD ANALYSIS
        st.markdown('<div class="section-header">Generalized Pareto Distribution (GPD) Analysis</div>', unsafe_allow_html=True)
        
        # Menghitung threshold untuk GPD
        threshold = np.percentile(returns_series, gpd_threshold_percentile)
        st.write(f"Threshold GPD (percentile {gpd_threshold_percentile}%): {threshold:.6f} ({threshold*100:.4f}%)")
        
        # Extract extreme values (dibawah threshold)
        extreme_values = returns_series[returns_series < threshold]
        count_extreme = len(extreme_values)
        
        st.write(f"Jumlah nilai ekstrem (di bawah threshold): {count_extreme} dari {len(returns_series)} observasi ({count_extreme/len(returns_series)*100:.2f}%)")
        
        # Sesuai GPD
        gpd_params = genpareto.fit(extreme_values)
        shape_gpd, loc_gpd, scale_gpd = gpd_params
        
        # Menghitung statistik GPD
        log_likelihood_gpd = np.sum(genpareto.logpdf(extreme_values, shape_gpd, loc_gpd, scale_gpd))
        AIC_gpd = 2 * 3 - 2 * log_likelihood_gpd  # 3 parameters
        
        # Menampilkan parameter GPD
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shape Parameter (ξ)", f"{shape_gpd:.6f}")
        with col2:
            st.metric("Location Parameter (μ)", f"{loc_gpd:.6f}")
        with col3:
            st.metric("Scale Parameter (σ)", f"{scale_gpd:.6f}")
        with col4:
            st.metric("AIC GPD", f"{AIC_gpd:.4f}")
        
        # GPD goodness-of-fit test
        ks_stat_gpd, p_value_gpd = kstest(extreme_values, "genpareto", args=(shape_gpd, loc_gpd, scale_gpd))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("KS Statistic GPD", f"{ks_stat_gpd:.4f}")
        with col2:
            st.metric("P-value GPD", f"{p_value_gpd:.4f}")
        
        if p_value_gpd > alpha:
            st.success("Distribusi GPD adalah fit yang baik untuk data ekstrem pada alpha 5% (H0 diterima).")
            gpd_good_fit = True
        else:
            st.warning("Distribusi GPD mungkin bukan fit yang baik untuk data ekstrem pada alpha 5% (H0 ditolak).")
            gpd_good_fit = False
        
        # Visualisasi GPD
        fig, ax = plt.subplots(figsize=(14, 6))
                                
        # Plot extreme values and threshold
        ax.plot(returns_series.values, 'b-', alpha=0.7, linewidth=0.8, label='Return Portofolio')
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold = {threshold*100:.4f}%')
        ax.scatter(np.where(returns_series < threshold)[0], extreme_values, color='red', s=25, alpha=0.8, label='Nilai Ekstrem')
        ax.set_xlabel('Indeks Data')
        ax.set_ylabel('Return')
        ax.set_title('Return Portofolio dan Threshold GPD\n(Titik Merah = Nilai Ekstrem untuk Analisis GPD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
                                
        plt.tight_layout()
        st.pyplot(fig)

        # VALUE AT RISK CALCULATION - HANYA GEV DAN GPD
        st.markdown('<div class="section-header">VALUE AT RISK (VaR) CALCULATION</div>', unsafe_allow_html=True)
        
        # Mengkonversikan tingkat kepercayaan menjadi probabilitas
        confidence_level = var_confidence / 100
        
        # Menghitung VaR dengan GEV

        # Fungsi untuk menghitung VaR Generalized Extreme Value 
        def calculate_gev_var(shape, loc, scale, confidence_level):
            term = -np.log(1 - confidence_level)
            
            if shape != 0:
                VaR = loc + (scale / shape) * ((term ** -shape) - 1)
            else:
                VaR = loc - scale * np.log(term)
            
            return VaR
        
        if gev_good_fit:
            gev_var = calculate_gev_var(shape_gev, loc_gev, scale_gev, confidence_level)
        else:
            gev_var = np.nan
        
        # Menghitung VaR dengan GPD
        if gpd_good_fit:
            # Untuk GPD, perlu menghitung VaR menggunakan probabilitas melebihi.
            p = 1 - confidence_level  # Probabilitas melampaui batas
            n = len(returns_series)
            nu = len(extreme_values)  # Jumlah pelanggaran
            gpd_var = threshold + (scale_gpd / shape_gpd) * (((n / nu) * p) ** (-shape_gpd) - 1)
        else:
            gpd_var = np.nan
        
        # Menampilkan hasil VaR
        st.markdown(f'<div class="var-result">', unsafe_allow_html=True)
        st.subheader(f"Value at Risk ({var_confidence}% Confidence Level) - 1 Hari")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not np.isnan(gev_var):
                st.metric("GEV VaR", f"{gev_var*100:.4f}%")
            else:
                st.metric("GEV VaR", "Tidak tersedia")
        with col2:
            if not np.isnan(gpd_var):
                st.metric("GPD VaR", f"{gpd_var*100:.4f}%")
            else:
                st.metric("GPD VaR", "Tidak tersedia")
        
        #  Menghitung VaR dalam satuan Rupiah
        st.subheader("VaR dalam Rupiah")
        
        if not np.isnan(gev_var):
            var_amount_gev = investment_amount * abs(gev_var)
            st.metric("GEV VaR", f"Rp {var_amount_gev:,.0f}")
        
        if not np.isnan(gpd_var):
            var_amount_gpd = investment_amount * abs(gpd_var)
            st.metric("GPD VaR", f"Rp {var_amount_gpd:,.0f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Rekomendasi berdasarkan AIC
        st.markdown('<div class="section-header">REKOMENDASI DISTRIBUSI UNTUK VaR</div>', unsafe_allow_html=True)
        
        if gev_good_fit and gpd_good_fit:
            if AIC_gev < AIC_gpd:
                st.success(f"**REKOMENDASI:** Gunakan GEV untuk perhitungan VaR (AIC GEV: {AIC_gev:.2f} < AIC GPD: {AIC_gpd:.2f})")
                recommended_var = gev_var
                recommended_method = "GEV"
            else:
                st.success(f"**REKOMENDASI:** Gunakan GPD untuk perhitungan VaR (AIC GPD: {AIC_gpd:.2f} < AIC GEV: {AIC_gev:.2f})")
                recommended_var = gpd_var
                recommended_method = "GPD"
        elif gev_good_fit:
            st.success(f"**REKOMENDASI:** Gunakan GEV untuk perhitungan VaR")
            recommended_var = gev_var
            recommended_method = "GEV"
        elif gpd_good_fit:
            st.success(f"**REKOMENDASI:** Gunakan GPD untuk perhitungan VaR")
            recommended_var = gpd_var
            recommended_method = "GPD"
        else:
            st.warning("**REKOMENDASI:** Tidak ada distribusi ekstrem yang cocok. Silakan sesuaikan parameter analisis.")
            recommended_var = None
            recommended_method = None
        
        if recommended_method in ["GEV", "GPD"]:
            st.info(f"""
            **Interpretasi VaR {recommended_method}:**
            - Dengan tingkat kepercayaan {var_confidence}%, kerugian harian terburuk tidak akan melebihi **{abs(recommended_var)*100:.4f}%**
            - Dalam nilai Rupiah: **Rp {investment_amount * abs(recommended_var):,.0f}**
            - Artinya, hanya ada {100-var_confidence}% kemungkinan mengalami kerugian lebih besar dari nilai tersebut dalam 1 hari
            """)
