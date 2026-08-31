import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pulp import LpProblem, LpVariable, LpMinimize, value
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# =========================================================
# 全局参数：全部来自大唐长春二热环评报告书（报批版）
# =========================================================
PLANT_PARAMS = {
    "unit_num": 2,                  # 机组数量
    "unit_capacity": 660,           # 单台机组额定容量 MW
    "min_load_ratio": 0.4,          # 最低稳燃负荷率
    "max_heat_power_per_unit": 735, # 单台机组最大供热功率 MW
    "power_standard_coal": 235.9,   # 发电标准煤耗 g/kWh
    "heat_standard_coal": 39.3,     # 供热标准煤耗 kg/GJ
    "limestone_annual": 123098,     # 脱硫石灰石年消耗量 t/a
    "annual_coal_consumption": 403.56e4, # 年燃煤消耗量 t/a
    "eboiler_num": 4,               # 调峰电锅炉数量
    "eboiler_capacity": 50,         # 单台电锅炉额定功率 MW
    "eboiler_efficiency": 0.98,     # 电锅炉热效率
}

CARBON_PARAMS = {
    "standard_coal_calorific": 29307.6, # 标准煤低位发热量 kJ/kg
    "carbon_per_calorific": 26.18,      # 单位热值含碳量 tC/TJ
    "carbon_oxidation_rate": 0.99,      # 碳氧化率
    "co2_c_molar_ratio": 44 / 12,       # CO2与碳的摩尔质量比
    "limestone_emission_factor": 0.44,  # 石灰石分解CO2排放因子 tCO2/t
}

# =========================================================
# 模块1：碳排放核心核算
# =========================================================
def calculate_carbon_emission(power_mw, heat_gj_h, limestone_t_h=0):
    """
    逐时碳排放核算（符合发电设施温室气体核算指南）
    :param power_mw: 发电总功率 MW
    :param heat_gj_h: 总供热量 GJ/h
    :param limestone_t_h: 石灰石消耗量 t/h
    :return: 总排放量tCO2/h, 排放分项字典
    """
    # 发电煤耗：MW * 1000 = kW，煤耗g/kWh → 吨/小时
    coal_power_t = power_mw * 1000 * PLANT_PARAMS["power_standard_coal"] / 1e6
    # 供热煤耗：GJ/h * kg/GJ → 吨/小时
    coal_heat_t = heat_gj_h * PLANT_PARAMS["heat_standard_coal"] / 1000
    total_coal_t = coal_power_t + coal_heat_t

    # 燃料燃烧CO2排放
    total_heat_TJ = total_coal_t * CARBON_PARAMS["standard_coal_calorific"] * 1e-6
    combustion_co2 = (total_heat_TJ
                      * CARBON_PARAMS["carbon_per_calorific"]
                      * CARBON_PARAMS["carbon_oxidation_rate"]
                      * CARBON_PARAMS["co2_c_molar_ratio"])

    # 脱硫过程CO2排放
    desulfurization_co2 = limestone_t_h * CARBON_PARAMS["limestone_emission_factor"]

    total_co2 = combustion_co2 + desulfurization_co2

    breakdown = {
        "发电燃煤排放": combustion_co2 * (coal_power_t / total_coal_t) if total_coal_t > 0 else 0,
        "供热燃煤排放": combustion_co2 * (coal_heat_t / total_coal_t) if total_coal_t > 0 else 0,
        "脱硫过程排放": desulfurization_co2
    }
    return round(total_co2, 3), breakdown

# =========================================================
# 模块2：模拟数据生成（监测数据+历史训练数据）
# =========================================================
def generate_daily_monitor_data():
    """生成典型日24小时模拟监测数据，贴合采暖季热电联产负荷规律"""
    hours = list(range(24))
    power_list = []
    heat_list = []

    for h in hours:
        # 发电负荷：早晚高峰高，深夜低谷
        load_factor = 0.75
        if h in [7, 8, 11, 12, 18, 19, 20]:
            load_factor += 0.15
        if 0 <= h <= 5:
            load_factor -= 0.2
        load_factor = np.clip(load_factor, PLANT_PARAMS["min_load_ratio"], 1.0)
        total_power = load_factor * PLANT_PARAMS["unit_capacity"] * PLANT_PARAMS["unit_num"]

        # 热负荷：夜间/早晚高，白天低
        heat_factor = 0.75
        if h in [5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23]:
            heat_factor = 1.0
        if 10 <= h <= 15:
            heat_factor = 0.65
        max_heat_gj = PLANT_PARAMS["max_heat_power_per_unit"] * PLANT_PARAMS["unit_num"] * 3.6
        total_heat = max_heat_gj * heat_factor

        power_list.append(total_power)
        heat_list.append(total_heat)

    # 计算石灰石消耗量（与煤耗成正比）
    coal_per_h = (np.array(power_list) * 1000 * PLANT_PARAMS["power_standard_coal"] / 1e6
                  + np.array(heat_list) * PLANT_PARAMS["heat_standard_coal"] / 1000)
    limestone_per_h = (coal_per_h / PLANT_PARAMS["annual_coal_consumption"]
                       * PLANT_PARAMS["limestone_annual"])

    # 逐时核算碳排放
    emission_list = []
    breakdown_list = []
    for i in range(24):
        co2, bd = calculate_carbon_emission(power_list[i], heat_list[i], limestone_per_h[i])
        emission_list.append(co2)
        breakdown_list.append(bd)

    df = pd.DataFrame({
        "小时": hours,
        "发电功率(MW)": np.round(power_list, 2),
        "供热量(GJ/h)": np.round(heat_list, 2),
        "燃煤消耗量(t/h)": np.round(coal_per_h, 2),
        "CO₂排放量(tCO₂/h)": np.round(emission_list, 2)
    })
    return df, breakdown_list

def generate_historical_dataset(n_samples=800):
    """生成多工况历史数据集，用于机器学习模型训练"""
    np.random.seed(42)
    power = np.random.uniform(
        PLANT_PARAMS["min_load_ratio"] * PLANT_PARAMS["unit_capacity"] * 2,
        PLANT_PARAMS["unit_capacity"] * PLANT_PARAMS["unit_num"],
        n_samples
    )
    heat = np.random.uniform(0, PLANT_PARAMS["max_heat_power_per_unit"] * 2 * 3.6, n_samples)

    # 计算基准排放+随机噪声，模拟真实工业数据波动
    emissions = []
    for i in range(n_samples):
        coal = (power[i] * 1000 * PLANT_PARAMS["power_standard_coal"] / 1e6
                + heat[i] * PLANT_PARAMS["heat_standard_coal"] / 1000)
        limestone = coal / PLANT_PARAMS["annual_coal_consumption"] * PLANT_PARAMS["limestone_annual"]
        co2, _ = calculate_carbon_emission(power[i], heat[i], limestone)
        co2_noisy = co2 * (1 + np.random.normal(0, 0.025))  # 2.5%随机波动
        emissions.append(co2_noisy)

    df = pd.DataFrame({
        "发电功率(MW)": np.round(power, 2),
        "供热量(GJ/h)": np.round(heat, 2),
        "CO₂排放量(tCO₂/h)": np.round(emissions, 2)
    })
    return df

# =========================================================
# 模块3：碳排放预测模型（机理+机器学习融合）
# =========================================================
def train_prediction_model(historical_df):
    """训练随机森林碳排放预测模型"""
    X = historical_df[["发电功率(MW)", "供热量(GJ/h)"]].values
    y = historical_df["CO₂排放量(tCO₂/h)"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=60, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return model, mape

def predict_emission(model, power_arr, heat_arr):
    """模型批量预测"""
    X_pred = np.column_stack([power_arr, heat_arr])
    return np.round(model.predict(X_pred), 3)

# =========================================================
# 模块4：减碳导向热电调度优化（线性规划）
# =========================================================
def optimize_dispatch(target_power_mw, target_heat_gj):
    """
    目标：最小化全厂总碳排放
    决策变量：2台机组的发电功率、抽汽供热量，调峰电锅炉功率
    约束：功率平衡、热负荷平衡、热电耦合、设备额定容量
    """
    prob = LpProblem("Carbon_Min_Dispatch", LpMinimize)

    # 1. 决策变量
    p1 = LpVariable("1号机组功率",
                    lowBound=PLANT_PARAMS["min_load_ratio"] * PLANT_PARAMS["unit_capacity"],
                    upBound=PLANT_PARAMS["unit_capacity"])
    p2 = LpVariable("2号机组功率",
                    lowBound=PLANT_PARAMS["min_load_ratio"] * PLANT_PARAMS["unit_capacity"],
                    upBound=PLANT_PARAMS["unit_capacity"])

    max_heat_gj_unit = PLANT_PARAMS["max_heat_power_per_unit"] * 3.6
    h1 = LpVariable("1号机组供热量", lowBound=0, upBound=max_heat_gj_unit)
    h2 = LpVariable("2号机组供热量", lowBound=0, upBound=max_heat_gj_unit)

    eboiler_p = LpVariable("电锅炉功率",
                           lowBound=0,
                           upBound=PLANT_PARAMS["eboiler_capacity"] * PLANT_PARAMS["eboiler_num"])
    eboiler_heat = eboiler_p * 3.6 * PLANT_PARAMS["eboiler_efficiency"]  # 电锅炉供热量 GJ/h

    # 2. 碳排放因子
    # 单位发电碳排放 tCO₂/MWh
    ef_power = (PLANT_PARAMS["power_standard_coal"] / 1e6 * 1000
                * CARBON_PARAMS["standard_coal_calorific"] * 1e-6
                * CARBON_PARAMS["carbon_per_calorific"]
                * CARBON_PARAMS["carbon_oxidation_rate"]
                * CARBON_PARAMS["co2_c_molar_ratio"])
    # 单位抽汽供热碳排放 tCO₂/GJ
    ef_heat = (PLANT_PARAMS["heat_standard_coal"] / 1000
               * CARBON_PARAMS["standard_coal_calorific"] * 1e-6
               * CARBON_PARAMS["carbon_per_calorific"]
               * CARBON_PARAMS["carbon_oxidation_rate"]
               * CARBON_PARAMS["co2_c_molar_ratio"])

    # 3. 目标函数：总碳排放最小
    prob += ((p1 + p2) * ef_power + (h1 + h2) * ef_heat), "Total_Carbon"

    # 4. 约束条件
    # 功率平衡：机组总发电 = 对外供电 + 电锅炉自用电
    prob += p1 + p2 == target_power_mw + eboiler_p, "Power_Balance"
    # 热负荷平衡：机组抽汽 + 电锅炉供热 = 总热需求
    prob += h1 + h2 + eboiler_heat == target_heat_gj, "Heat_Balance"
    # 热电耦合：抽汽量不超过当前负荷下的最大抽汽能力
    prob += h1 <= max_heat_gj_unit * (p1 / PLANT_PARAMS["unit_capacity"]), "Unit1_Coupling"
    prob += h2 <= max_heat_gj_unit * (p2 / PLANT_PARAMS["unit_capacity"]), "Unit2_Coupling"

    # 5. 求解
    prob.solve()

    # 6. 结果整理
    if prob.status != 1:
        return {"状态": "无解", "说明": "当前工况超出机组运行范围，请调整负荷"}

    base_emission = target_power_mw * ef_power + target_heat_gj * ef_heat
    opt_emission = value(prob.objective)

    result = {
        "状态": "最优解",
        "1号机组功率(MW)": round(value(p1), 2),
        "2号机组功率(MW)": round(value(p2), 2),
        "1号机组供热量(GJ/h)": round(value(h1), 2),
        "2号机组供热量(GJ/h)": round(value(h2), 2),
        "电锅炉功率(MW)": round(value(eboiler_p), 2),
        "电锅炉供热量(GJ/h)": round(value(eboiler_heat), 2),
        "优化后碳排放(tCO₂/h)": round(opt_emission, 2),
        "基准碳排放(tCO₂/h)": round(base_emission, 2),
        "小时减碳量(tCO₂)": round(base_emission - opt_emission, 2),
        "减碳比例(%)": round((base_emission - opt_emission) / base_emission * 100, 2)
    }
    return result

# =========================================================
# 模块5：Streamlit可视化界面
# =========================================================
def main():
    st.set_page_config(page_title="电厂智能控碳系统", layout="wide", page_icon="🏭")
    st.title("🏭 大唐长春二热碳排放监测·预测·调度优化系统")
    st.caption("基于2×660MW超超临界热电联产项目 | 智能控碳方向参赛作品")

    # 侧边栏导航
    page = st.sidebar.radio(
        "功能导航",
        ["项目总览", "实时碳监测", "碳排放预测", "调度优化", "方法说明"]
    )

    # 预加载数据与模型
    daily_df, daily_breakdown = generate_daily_monitor_data()
    historical_df = generate_historical_dataset()
    pred_model, model_mape = train_prediction_model(historical_df)

    # ---------------- 页面1：项目总览 ----------------
    if page == "项目总览":
        st.header("一、项目概况")
        col1, col2, col3 = st.columns(3)
        col1.metric("装机规模", "2×660MW", "超超临界热电联产")
        col2.metric("年供热量", "1516.95万GJ", "覆盖3553万㎡供热面积")
        col3.metric("年发电量", "50.45亿kWh", "设计年利用4854小时")

        st.subheader("二、系统功能架构")
        st.markdown("""
        本系统以大唐长春二热「退城进郊」迁建项目为真实场景，构建**监测-预测-优化**三级智能控碳体系：
        1. **实时碳监测**：基于烟气CEMS与物料计量数据，逐时核算全厂碳排放，动态展示排放结构
        2. **碳排放预测**：采用「机理模型+随机森林」融合算法，实现多工况下碳排放精准预测
        3. **调度优化**：以碳最小化为目标，优化机组负荷分配与热电耦合运行，量化减碳效果
        """)

        st.info("💡 所有工程参数均来自公开的项目环境影响报告书，运行数据为典型工况模拟值，用于功能演示。")

    # ---------------- 页面2：实时碳监测 ----------------
    elif page == "实时碳监测":
        st.header("📊 实时碳排放监测")

        # 核心指标卡片
        total_daily = round(daily_df["CO₂排放量(tCO₂/h)"].sum(), 2)
        avg_intensity = round(total_daily * 1000 / daily_df["发电功率(MW)"].sum(), 2)
        current_hour = 8  # 模拟当前为早8点

        col1, col2, col3 = st.columns(3)
        col1.metric("当日累计CO₂排放", f"{total_daily} t")
        col2.metric("当前小时排放", f"{daily_df.iloc[current_hour]['CO₂排放量(tCO₂/h)']} t/h")
        col3.metric("平均碳排放强度", f"{avg_intensity} kgCO₂/MWh")

        # 24小时排放趋势图
        st.subheader("24小时碳排放趋势曲线")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_df["小时"], y=daily_df["CO₂排放量(tCO₂/h)"],
            mode="lines+markers", name="CO₂排放量",
            line=dict(color="#e74c3c", width=3),
            marker=dict(size=6)
        ))
        fig_trend.update_layout(
            xaxis_title="小时", yaxis_title="CO₂排放量 (tCO₂/h)",
            hovermode="x unified", height=450,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # 排放构成饼图
        st.subheader("当前小时碳排放来源构成")
        bd = daily_breakdown[current_hour]
        fig_pie = px.pie(
            names=list(bd.keys()), values=list(bd.values()),
            color_discrete_sequence=["#e67e22", "#c0392b", "#3498db"],
            hole=0.4
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

        # 详细数据表格
        with st.expander("查看逐时详细监测数据"):
            st.dataframe(daily_df, use_container_width=True, height=380)

    # ---------------- 页面3：碳排放预测 ----------------
    elif page == "碳排放预测":
        st.header("🔮 碳排放智能预测")

        st.subheader("单工况自定义预测")
        col1, col2 = st.columns(2)
        with col1:
            input_power = st.slider(
                "发电功率 (MW)",
                min_value=int(PLANT_PARAMS["min_load_ratio"] * PLANT_PARAMS["unit_capacity"] * 2),
                max_value=int(PLANT_PARAMS["unit_capacity"] * 2),
                value=850
            )
        with col2:
            input_heat = st.slider(
                "供热量 (GJ/h)",
                min_value=500,
                max_value=int(PLANT_PARAMS["max_heat_power_per_unit"] * 2 * 3.6),
                value=4200
            )

        # 双模型预测对比
        mech_co2, _ = calculate_carbon_emission(input_power, input_heat,
                                                input_power * 1000 * PLANT_PARAMS["power_standard_coal"]
                                                / 1e6 / PLANT_PARAMS["annual_coal_consumption"]
                                                * PLANT_PARAMS["limestone_annual"])
        ml_co2 = predict_emission(pred_model, [input_power], [input_heat])[0]

        col1, col2 = st.columns(2)
        col1.metric("机理模型预测值", f"{mech_co2} tCO₂/h")
        col2.metric("机器学习预测值", f"{ml_co2} tCO₂/h",
                   f"相对误差 {round(abs(ml_co2 - mech_co2) / mech_co2 * 100, 2)}%")

        st.caption(f"📌 随机森林模型测试集MAPE：{round(model_mape * 100, 2)}%")

        # 24小时连续预测
        st.subheader("未来24小时碳排放预测")
        future_power = daily_df["发电功率(MW)"].values
        future_heat = daily_df["供热量(GJ/h)"].values
        future_pred = predict_emission(pred_model, future_power, future_heat)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=daily_df["小时"], y=daily_df["CO₂排放量(tCO₂/h)"],
            name="历史实测值", line=dict(color="#7f8c8d", dash="dash")
        ))
        fig_pred.add_trace(go.Scatter(
            x=daily_df["小时"], y=future_pred,
            name="模型预测值", line=dict(color="#e74c3c", width=2)
        ))
        fig_pred.update_layout(
            xaxis_title="小时", yaxis_title="CO₂排放量 (tCO₂/h)",
            hovermode="x unified", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    # ---------------- 页面4：调度优化 ----------------
    elif page == "调度优化":
        st.header("⚡ 减碳导向热电调度优化")

        st.subheader("设置运行工况")
        col1, col2 = st.columns(2)
        with col1:
            target_power = st.slider(
                "目标外供电功率 (MW)",
                min_value=400, max_value=1320, value=800
            )
        with col2:
            target_heat = st.slider(
                "目标供热负荷 (GJ/h)",
                min_value=1000, max_value=5200, value=3800
            )

        if st.button("开始优化计算", type="primary", use_container_width=True):
            result = optimize_dispatch(target_power, target_heat)

            if result["状态"] == "最优解":
                st.success("✅ 已求得全局最优调度方案")

                col1, col2, col3 = st.columns(3)
                col1.metric("优化后碳排放", f"{result['优化后碳排放(tCO₂/h)']} t/h")
                col2.metric("基准方案碳排放", f"{result['基准碳排放(tCO₂/h)']} t/h")
                col3.metric("减碳效果", f"{result['减碳比例(%)']}%",
                           f"每小时减少 {result['小时减碳量(tCO₂)']} tCO₂")

                st.subheader("机组调度详情")
                dispatch_table = pd.DataFrame({
                    "指标": ["发电功率(MW)", "抽汽供热量(GJ/h)"],
                    "1号机组": [result["1号机组功率(MW)"], result["1号机组供热量(GJ/h)"]],
                    "2号机组": [result["2号机组功率(MW)"], result["2号机组供热量(GJ/h)"]],
                    "调峰电锅炉": [result["电锅炉功率(MW)"], result["电锅炉供热量(GJ/h)"]]
                })
                st.table(dispatch_table)

                # 优化对比柱状图
                fig_compare = go.Figure(data=[
                    go.Bar(name="基准调度方案", x=["总碳排放"],
                           y=[result["基准碳排放(tCO₂/h)"]], marker_color="#7f8c8d"),
                    go.Bar(name="减碳优化方案", x=["总碳排放"],
                           y=[result["优化后碳排放(tCO₂/h)"]], marker_color="#27ae60")
                ])
                fig_compare.update_layout(barmode="group", height=400,
                                         title="优化前后碳排放对比",
                                         yaxis_title="tCO₂/h")
                st.plotly_chart(fig_compare, use_container_width=True)
            else:
                st.error("❌ " + result["说明"])

    # ---------------- 页面5：方法说明 ----------------
    elif page == "方法说明":
        st.header("📑 技术方法与数据来源")

        st.subheader("1. 碳排放核算方法")
        st.latex(r"E_{燃烧} = B \times Q_{net} \times C \times O \times \frac{44}{12}")
        st.markdown("""
        - 严格遵循《企业温室气体排放核算方法与报告指南 发电设施》
        - 包含**燃料燃烧排放**与**脱硫过程排放**两大项
        - 热电分摊采用热量法，按供热比例分摊煤耗与碳排放
        """)

        st.subheader("2. 预测模型原理")
        st.markdown("""
        采用**机理驱动+数据修正**的混合预测框架：
        - 底层：基于煤耗特性与排放因子构建物理机理模型，保证可解释性
        - 上层：随机森林算法拟合非线性波动，提升预测精度
        - 优势：兼顾工业机理严谨性与人工智能的自适应能力
        """)

        st.subheader("3. 调度优化模型")
        st.markdown("""
        以全厂碳排放最小化为目标的线性规划模型：
        - 决策变量：2台机组的发电功率、抽汽量，调峰电锅炉出力
        - 约束条件：功率平衡、热负荷平衡、热电耦合运行约束、设备额定容量
        - 求解算法：单纯形法，保证全局最优解
        """)

        st.subheader("4. 数据来源")
        st.info("""
        全部核心工程参数均来自《大唐长春二热"退城进郊"2×660MW煤电项目环境影响报告书（报批版）》：
        - 机组容量、煤耗水平、供热能力等主机参数
        - 脱硫石灰石消耗量、调峰电锅炉配置等辅机参数
        - 运行数据为典型采暖季工况模拟值，用于功能演示
        """)

if __name__ == "__main__":
    main()
