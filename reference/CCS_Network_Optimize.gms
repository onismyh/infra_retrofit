*============================*
* 动态 CCS 网络优化模型源码 *
*============================*

* 引入外部数据
$include ChinaCCS_include_V1.txt

*=============================================*
* 基础参数定义（时间、贴现、生命周期等）
*=============================================*
set t "年份集合";

scalar
  base_year /2025/             
  discount_rate /0.08/          
  Project_Lifetime /30/;        

parameter year_idx(t) 
/
2035  2035
2060  2060
/;

parameter discount_factor(t) "CAPEX贴现因子";
loop(t,
  discount_factor(t) = 1 / power(1 + discount_rate, year_idx(t) - base_year);
);

scalar opex_discount_factor "OPEX贴现因子";
opex_discount_factor = (1 / discount_rate) * (1 - 1 / ((1 + discount_rate) ** Project_Lifetime));

*=============================================*
* 决策变量定义
*=============================================*

positive variables
*=== CO2 捕集量（按行业细分，new / stock）===
CO2_capture_power_new(sources,t)
CO2_capture_power_stock(sources,t)

CO2_capture_cem_new(sources,t)
CO2_capture_cem_stock(sources,t)

CO2_capture_isi_new(sources,t)
CO2_capture_isi_stock(sources,t)

CO2_capture_chemical_new(sources,t)
CO2_capture_chemical_stock(sources,t)

CO2_capture_nh3_new(sources,t)
CO2_capture_nh3_stock(sources,t)

CO2_capture_meoh_new(sources,t)
CO2_capture_meoh_stock(sources,t)

CO2_capture_ref_new(sources,t)
CO2_capture_ref_stock(sources,t)

CO2_capture_liquid_new(sources,t)
CO2_capture_liquid_stock(sources,t)

CO2_capture_ngas_new(sources,t)
CO2_capture_ngas_stock(sources,t)

CO2_capture_olefin_new(sources,t)
CO2_capture_olefin_stock(sources,t)

CO2_capture_glycol_new(sources,t)
CO2_capture_glycol_stock(sources,t)

CO2_capture_total_new(sources,t)
CO2_capture_total_stock(sources,t)

*=== CO2 封存量===
CO2_inject_new(sinks,t)
CO2_inject_stock(sinks,t)

*=== CO2 运输量=== 
CO2_flow_new(node_from,node_to,t) 
CO2_flow_stock(node_from,node_to,t)

*=== CO2 管道长度===
Total_CO2_Pipeline_Length(t)  

*=== 成本变量 ===
CAPEX_Capture(t)
OPEX_Capture(t)
Cost_Capture(t)

CAPEX_Storage(t)
OPEX_Storage(t)
Cost_Storage(t)

Route_CAPEX_Transport(node_from,node_to,t)
CAPEX_Transport(t)
OPEX_Transport(t)
Cost_Transport(t)

Revenue_EOR(t);

variable objective;

integer variables
*=== 注入井与产品井数量（new / stock）===
num_inject_well_new(sinks,t)
num_inject_well_stock(sinks,t)
  
num_total_well_new(sinks,t)

*===管道数量===
num_pipe_new(node_from,node_to,d,t)  
num_pipe_stock(node_from,node_to,d,t) 
num_pipes_total(node_from,node_to,t);

*===变量约束======
CO2_flow_stock.up(node_from,node_to,t)=100;
num_pipes_total.up(node_from,node_to,t)=2;


*=============================================*
* 方程定义
*=============================================*
equations
*=== 捕集量累计方程及上限约束声明 =================
Eqa_capture_power_init(sources)
Eqa_capture_power_cum(sources,t)

Eqa_capture_cem_init(sources)
Eqa_capture_cem_cum(sources,t)

Eqa_capture_isi_init(sources)
Eqa_capture_isi_cum(sources,t)

Eqa_capture_chemical_cum(sources, t)

Eqa_capture_nh3_init(sources)
Eqa_capture_nh3_cum(sources, t)

Eqa_capture_meoh_init(sources)
Eqa_capture_meoh_cum(sources, t)
    
Eqa_capture_ref_init(sources)
Eqa_capture_ref_cum(sources, t)
    
Eqa_capture_liquid_init(sources)
Eqa_capture_liquid_cum(sources, t)
    
Eqa_capture_ngas_init(sources)
Eqa_capture_ngas_cum(sources, t)
    
Eqa_capture_olefin_init(sources)
Eqa_capture_olefin_cum(sources, t)
    
Eqa_capture_glycol_init(sources)
Eqa_capture_glycol_cum(sources, t)

Eqa_capture_total(sources,t)
Eqa_capture_total_init(sources)
Eqa_capture_total_cum(sources, t)

constraint_CO2_capture_power_limit(sources,t)
constraint_CO2_capture_cem_limit(sources,t)
constraint_CO2_capture_isi_limit(sources,t)
constraint_CO2_capture_nh3_limit(sources,t)
constraint_CO2_capture_meoh_limit(sources,t)
constraint_CO2_capture_ref_limit(sources,t)
constraint_CO2_capture_liquid_limit(sources,t)
constraint_CO2_capture_ngas_limit(sources,t)
constraint_CO2_capture_olefin_limit(sources,t)
constraint_CO2_capture_glycol_limit(sources,t)

*=== 封存量累计方程及井口注入容量声明 =======================================================
Eqa_inject_CO2_init(sinks)
Eqa_inject_CO2_cum(sinks,t)

Eqa_numwell_init(sinks)
Eqa_numwell_cum(sinks,t)

Eqa_inject_numwell(sinks,t)
Eqa_Total_numwell(sinks,t)

constraint_inject_limit(sinks,t)

*=== 运输量及管道数量累计方程和连通性约束声明 ======================================
Eqa_CO2_mass_conservation_sources(sources,t)
Eqa_CO2_mass_conservation_sinks(sinks,t)
Eqa_CO2flow_init(node_from,node_to)
Eqa_CO2flow_cum(node_from,node_to,t)

Eqa_CO2pipe_init(node_from,node_to,d)
Eqa_CO2pipe_cum(node_from,node_to,d,t)
Eqa_CO2pipe_total(node_from,node_to,t)

constraint_CO2pipe_capacity_upper(node_from,node_to,t)
*constraint_CO2pipe_capacity_lower(node_from,node_to,t)

constraint_CO2flow_1(node_from,node_to,t)
constraint_CO2flow_2(node_from,node_to,t)

Eqa_Total_CO2_Pipeline_Length(t)

*==================== 成本计算方程声明 =============================================
Eqa_CAPEX_Capture(t)
Eqa_OPEX_Capture(t)
Eqa_Cost_Capture(t)

Eqa_CAPEX_Storage(t)
Eqa_OPEX_Storage(t)
Eqa_Cost_Storage(t)

Eqa_Route_CAPEX_Transport(node_from,node_to,t)
Eqa_CAPEX_Transport(t)
Eqa_OPEX_Transport(t)
Eqa_Cost_Transport(t)

Eqa_Revenue_EOR(t)

*==================== CO2捕集量政策目标 =============================================
Eqa_Target_CO2_power(t)
Eqa_Target_CO2_isi(t)
Eqa_Target_CO2_cem(t)
Eqa_Target_CO2_chemical(t)

*========================= 目标函数 ===============================================
Eqa_objective;

*=================================================================================
*捕集环节方程
*=================================================================================
*=================捕集量动态化=====================
Eqa_capture_power_init(sources)..  CO2_capture_power_stock(sources,'2035') =e= CO2_capture_power_new(sources,'2035');
Eqa_capture_power_cum(sources,t)$(ord(t)>1).. CO2_capture_power_stock(sources,t) =e= CO2_capture_power_stock(sources,t-1) + CO2_capture_power_new(sources,t);

Eqa_capture_cem_init(sources).. CO2_capture_cem_stock(sources,'2035') =e= CO2_capture_cem_new(sources,'2035');
Eqa_capture_cem_cum(sources,t)$(ord(t)>1).. CO2_capture_cem_stock(sources,t) =e= CO2_capture_cem_stock(sources,t-1) + CO2_capture_cem_new(sources,t);

Eqa_capture_isi_init(sources)..    CO2_capture_isi_stock(sources,'2035') =e= CO2_capture_isi_new(sources,'2035');
Eqa_capture_isi_cum(sources,t)$(ord(t)>1).. CO2_capture_isi_stock(sources,t) =e= CO2_capture_isi_stock(sources,t-1) + CO2_capture_isi_new(sources,t);

Eqa_capture_nh3_init(sources).. 
    CO2_capture_nh3_stock(sources, '2035') =e= CO2_capture_nh3_new(sources, '2035');

Eqa_capture_nh3_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_nh3_stock(sources, t) =e= CO2_capture_nh3_stock(sources, t-1) + CO2_capture_nh3_new(sources, t);

Eqa_capture_meoh_init(sources).. 
    CO2_capture_meoh_stock(sources, '2035') =e= CO2_capture_meoh_new(sources, '2035');

Eqa_capture_meoh_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_meoh_stock(sources, t) =e= CO2_capture_meoh_stock(sources, t-1) + CO2_capture_meoh_new(sources, t);

Eqa_capture_ref_init(sources).. 
    CO2_capture_ref_stock(sources, '2035') =e= CO2_capture_ref_new(sources, '2035');

Eqa_capture_ref_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_ref_stock(sources, t) =e= CO2_capture_ref_stock(sources, t-1) + CO2_capture_ref_new(sources, t);

Eqa_capture_liquid_init(sources).. 
    CO2_capture_liquid_stock(sources, '2035') =e= CO2_capture_liquid_new(sources, '2035');

Eqa_capture_liquid_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_liquid_stock(sources, t) =e= CO2_capture_liquid_stock(sources, t-1) + CO2_capture_liquid_new(sources, t);

Eqa_capture_ngas_init(sources).. 
    CO2_capture_ngas_stock(sources, '2035') =e= CO2_capture_ngas_new(sources, '2035');

Eqa_capture_ngas_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_ngas_stock(sources, t) =e= CO2_capture_ngas_stock(sources, t-1) + CO2_capture_ngas_new(sources, t);

Eqa_capture_olefin_init(sources).. 
    CO2_capture_olefin_stock(sources, '2035') =e= CO2_capture_olefin_new(sources, '2035');

Eqa_capture_olefin_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_olefin_stock(sources, t) =e= CO2_capture_olefin_stock(sources, t-1) + CO2_capture_olefin_new(sources, t);

Eqa_capture_glycol_init(sources).. 
    CO2_capture_glycol_stock(sources, '2035') =e= CO2_capture_glycol_new(sources, '2035');

Eqa_capture_glycol_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_glycol_stock(sources, t) =e= CO2_capture_glycol_stock(sources, t-1) + CO2_capture_glycol_new(sources, t);
    
Eqa_capture_chemical_cum(sources, t).. 
    CO2_capture_chemical_stock(sources, t) =e= CO2_capture_nh3_stock(sources,t)+
    CO2_capture_meoh_stock(sources,t)+
    CO2_capture_ref_stock(sources,t)+
    CO2_capture_liquid_stock(sources,t) +
    CO2_capture_ngas_stock(sources,t)+
    CO2_capture_olefin_stock(sources,t)+
    CO2_capture_glycol_stock(sources,t);

*======================节点各行业捕集量加总==================================================
Eqa_capture_total(sources,t).. CO2_capture_total_new(sources,t) =e=
    CO2_capture_power_new(sources,t) +
    CO2_capture_cem_new(sources,t) +
    CO2_capture_isi_new(sources,t) +
    CO2_capture_nh3_new(sources,t)+
    CO2_capture_meoh_new(sources,t)+
    CO2_capture_ref_new(sources,t)+
    CO2_capture_liquid_new(sources,t) +
    CO2_capture_ngas_new(sources,t)+
    CO2_capture_olefin_new(sources,t)+
    CO2_capture_glycol_new(sources,t);

Eqa_capture_total_init(sources).. 
    CO2_capture_total_stock(sources, '2035') =e= CO2_capture_total_new(sources, '2035');

Eqa_capture_total_cum(sources, t)$(ord(t) > 1).. 
    CO2_capture_total_stock(sources, t) =e= CO2_capture_total_stock(sources, t-1) + CO2_capture_total_new(sources, t);

*=== 各行业 stock变量 ≤ 各节点行业可捕集量上限 ============================================
constraint_CO2_capture_power_limit(sources,t)..
    CO2_capture_power_stock(sources,t) =l= Data_Source(sources, "CO2_power")*capture_factor;

constraint_CO2_capture_cem_limit(sources,t)..
    CO2_capture_cem_stock(sources,t) =l= Data_Source(sources, "CO2_cement")*capture_factor;

constraint_CO2_capture_isi_limit(sources,t)..
    CO2_capture_isi_stock(sources,t) =l= Data_Source(sources, "CO2_isi")*capture_factor;

constraint_CO2_capture_nh3_limit(sources,t)..
    CO2_capture_nh3_stock(sources,t) =l= Data_Source(sources, "CO2_ammonia")*capture_factor;

constraint_CO2_capture_meoh_limit(sources,t)..
    CO2_capture_meoh_stock(sources,t) =l= Data_Source(sources, "CO2_methanol")*capture_factor;

constraint_CO2_capture_ref_limit(sources,t)..
    CO2_capture_ref_stock(sources,t) =l= Data_Source(sources, "CO2_refinery")*capture_factor;

constraint_CO2_capture_liquid_limit(sources,t)..
    CO2_capture_liquid_stock(sources,t) =l= Data_Source(sources, "CO2_liquid")*capture_factor;

constraint_CO2_capture_ngas_limit(sources,t)..
    CO2_capture_ngas_stock(sources,t) =l= Data_Source(sources, "CO2_syngas")*capture_factor;

constraint_CO2_capture_olefin_limit(sources,t)..
    CO2_capture_olefin_stock(sources,t) =l= Data_Source(sources, "CO2_olefin")*capture_factor;

constraint_CO2_capture_glycol_limit(sources,t)..
    CO2_capture_glycol_stock(sources,t) =l= Data_Source(sources, "CO2_glycol")*capture_factor;

*================================================================================
*CO2封存环节方程定义
*================================================================================
*========================封存量动态化 =============================
Eqa_inject_CO2_init(sinks).. CO2_inject_stock(sinks,'2035') =e= CO2_inject_new(sinks,'2035');
Eqa_inject_CO2_cum(sinks,t)$(ord(t)>1).. CO2_inject_stock(sinks,t) =e= CO2_inject_stock(sinks,t-1) + CO2_inject_new(sinks,t);

Eqa_numwell_init(sinks).. num_inject_well_stock(sinks,'2035') =e= num_inject_well_new(sinks,'2035');
Eqa_numwell_cum(sinks,t)$(ord(t)>1).. num_inject_well_stock(sinks,t) =e= num_inject_well_stock(sinks,t-1)+num_inject_well_new(sinks,t);

*==================井口数量及注入能力约束=============================
Eqa_inject_numwell(sinks,t).. num_inject_well_stock(sinks,t)*Data_Sink(sinks,"Injectivity") =g= CO2_inject_stock(sinks,t);

Eqa_Total_numwell(sinks,t).. num_total_well_new(sinks,t) =e= num_inject_well_new(sinks,t)*(1+Data_Sink(sinks,"Ratio_NumWellProduct2NumWellInject"));

constraint_inject_limit(sinks,t).. Data_Sink(sinks,"Sink_Capacity")*Data_Sink(sinks,"Efficiency_Factor") =g= CO2_inject_stock(sinks,t)*Project_Lifetime;

*================================================================================
*CO2运输环节方程定义
*================================================================================
* 节点CO2流量守恒方程
Eqa_CO2_mass_conservation_sources(sources,t)..
    sum(node_from, CO2_flow_stock(node_from,sources,t)) - sum(node_to,CO2_flow_stock(sources,node_to,t)) + CO2_capture_total_stock(sources,t)=e=0;

Eqa_CO2_mass_conservation_sinks(sinks,t)..
    sum(node_from, CO2_flow_stock(node_from,sinks,t)) - sum(node_to,CO2_flow_stock(sinks,node_to,t)) - CO2_inject_stock(sinks,t)=e=0;
    
*=== CO2运输量及管道数量累计 =======================================================
Eqa_CO2flow_init(node_from,node_to).. 
    CO2_flow_stock(node_from,node_to,'2035') =e= CO2_flow_new(node_from,node_to,'2035');

Eqa_CO2flow_cum(node_from,node_to,t)$(ord(t)>1).. 
    CO2_flow_stock(node_from,node_to,t) =e= CO2_flow_stock(node_from,node_to,t-1) + CO2_flow_new(node_from,node_to,t);

Eqa_CO2pipe_init(node_from,node_to,d).. 
    num_pipe_stock(node_from,node_to,d,'2035') =e= num_pipe_new(node_from,node_to,d,'2035');

Eqa_CO2pipe_cum(node_from,node_to,d,t)$(ord(t) > 1).. 
    num_pipe_stock(node_from,node_to,d,t) =e= num_pipe_stock(node_from,node_to,d,t-1) + num_pipe_new(node_from,node_to,d,t);

Eqa_CO2pipe_total(node_from,node_to,t).. 
    num_pipes_total(node_from,node_to,t) =e= sum(d, num_pipe_stock(node_from,node_to,d,t));

*=== CO2运输量与管道数量的关系 ===
constraint_CO2pipe_capacity_upper(node_from,node_to,t).. 
    sum(d, num_pipe_stock(node_from,node_to,d,t) * data_CO2pipe(d,"flow_Max")) =g= CO2_flow_stock(node_from,node_to,t);

*constraint_CO2pipe_capacity_lower(node_from,node_to,t).. 
*    sum(d, num_pipe_stock(node_from,node_to,d,t) * data_CO2pipe(d,"flow_Min")) =l= CO2_flow_stock(node_from,node_to,t);

* ===CO2管道流量约束，考察管线长度和连通性===
constraint_CO2flow_1(node_from,node_to,t)..
    CO2_flow_stock(node_from,node_to,t)$(Dist_matrix(node_from,node_to) > Interval_max) =e= 0;

constraint_CO2flow_2(node_from,node_to,t)..
    CO2_flow_stock(node_from,node_to,t)$(connectivity_matrix(node_from,node_to)=0) =e= 0;

Eqa_Total_CO2_Pipeline_Length(t)..
    Total_CO2_Pipeline_Length(t) =e= sum((node_from,node_to),num_pipes_total(node_from,node_to,t)*Dist_matrix(node_from,node_to));

*================================================================================
*成本计算
*================================================================================
* =====CO2捕集环节, 原始数据即为美元/吨============
Eqa_CAPEX_Capture(t).. CAPEX_Capture(t) =e= sum(sources, CO2_capture_total_new(sources,t) * ccs_cost(t,"CAPEX_Capture_unit"));

Eqa_OPEX_Capture(t)..
  OPEX_Capture(t) =e=
   sum(sources,
         CO2_capture_power_new(sources,t)   * ccs_cost(t,"OPEX_power_unit") +
         CO2_capture_cem_new(sources,t)     * ccs_cost(t,"OPEX_cem_unit") +
         CO2_capture_isi_new(sources,t)     * ccs_cost(t,"OPEX_isi_unit") +
         CO2_capture_nh3_new(sources,t)     * ccs_cost(t,"OPEX_nh3_unit") +
         CO2_capture_meoh_new(sources,t)    * ccs_cost(t,"OPEX_meoh_unit") +
         CO2_capture_ref_new(sources,t)     * ccs_cost(t,"OPEX_ref_unit") +
         CO2_capture_liquid_new(sources,t)  * ccs_cost(t,"OPEX_liquid_unit") +
         CO2_capture_ngas_new(sources,t)    * ccs_cost(t,"OPEX_ngas_unit") +
         CO2_capture_olefin_new(sources,t)  * ccs_cost(t,"OPEX_olefin_unit") +
         CO2_capture_glycol_new(sources,t)  * ccs_cost(t,"OPEX_glycol_unit")
       )
    * opex_discount_factor;
    
Eqa_Cost_Capture(t)..  Cost_Capture(t)  =e= CAPEX_Capture(t) + OPEX_Capture(t);

* ========CO2封存及产品收益环节,原始数据为CNY,化为百万美元=================
Eqa_CAPEX_Storage(t)..
    CAPEX_Storage(t) =e= sum(sinks,
                               num_total_well_new(sinks,t)*UnitCAPEX_Well_Construct*Data_Sink(sinks,"Sink_Depth")
                               )/1000000/6.8;
Eqa_OPEX_Storage(t)..
    OPEX_Storage(t) =e= sum(sinks,
                              CO2_inject_new(sinks,t) * Data_Sink(sinks,"Cost_OM") +
                              num_total_well_new(sinks,t)*UnitCAPEX_Well_Construct/1000000*Data_Sink(sinks,"Sink_Depth")*opex_rate_DSA*(Data_Sink(sinks,"Cost_OM")=0)+
                              Monitor_cost * CO2_inject_new(sinks,t)
                              )*opex_discount_factor/6.8;
  
Eqa_Cost_Storage(t)..  Cost_Storage(t) =e= CAPEX_Storage(t) + OPEX_Storage(t);

Eqa_Revenue_EOR(t)..   Revenue_EOR(t) =e= sum(sinks, CO2_inject_new(sinks,t)*Data_Sink(sinks,"RatioOutput2InjectedCO2"))*opex_discount_factor/6.8;

* ========CO2运输环节=================
* CO2管道建设成本为 万元人民币/km，转化成百万美元/km
Eqa_Route_CAPEX_Transport(node_from,node_to,t)..
    Route_CAPEX_Transport(node_from,node_to,t) =e=
      sum(d,
             num_pipe_new(node_from,node_to,d,t)*data_CO2pipe(d,"cost")/100+
             Price_LandCompensation*15*data_CO2pipe(d,"bandwidth" )*num_pipe_new(node_from,node_to,d,t)
             )/0.55*1.1*length_weighted(node_from,node_to)/6.8;
             
Eqa_CAPEX_Transport(t)..
    CAPEX_Transport(t) =e= sum((node_from,node_to),Route_CAPEX_Transport(node_from,node_to,t));

Eqa_OPEX_Transport(t)..
    OPEX_Transport(t) =e= sum((node_from,node_to), Route_CAPEX_Transport(node_from,node_to,t)* OPEX_CO2_Transport_rate * opex_discount_factor);

Eqa_Cost_Transport(t).. Cost_Transport(t) =e= CAPEX_Transport(t) + OPEX_Transport(t);

*================================================================================
*政策目标约束
*================================================================================
Eqa_Target_CO2_power(t)..   sum(sources, CO2_capture_power_stock(sources,t)) =g= Target_CO2_power(t);

Eqa_Target_CO2_isi(t)..     sum(sources, CO2_capture_isi_stock(sources,t))   =g= Target_CO2_isi(t);

Eqa_Target_CO2_cem(t)..     sum(sources, CO2_capture_cem_stock(sources,t))   =g= Target_CO2_cem(t);

Eqa_Target_CO2_chemical(t)..
    sum(sources,
         CO2_capture_nh3_stock(sources,t)+
         CO2_capture_meoh_stock(sources,t)+
         CO2_capture_ref_stock(sources,t)+
         CO2_capture_liquid_stock(sources,t)+
         CO2_capture_ngas_stock(sources,t)+
         CO2_capture_olefin_stock(sources,t)+
         CO2_capture_glycol_stock(sources,t)) =g= Target_CO2_chemical(t);

*=== 总目标函数 ===
Eqa_objective.. objective =e= sum(t, discount_factor(t) * (Cost_Capture(t) + Cost_Storage(t) + Cost_Transport(t)- Revenue_EOR(t)));

*=== 模型求解 ===
model Dynamic_CCS /all/;
option mip = gurobi;
option threads = -1;
option reslim=5000;
option OptCR=0.005;
solve Dynamic_CCS using mip minimizing objective;

execute_unload "ccs_output.gdx";

*=== 输出结果到excel ===
$setglobal outputfile ChinaCCS.xlsm

$libinclude xldump CO2_capture_power_stock.l      %outputfile% CO2_capture_power_stock!
$libinclude xldump CO2_capture_cem_stock.l     %outputfile% CO2_capture_cem_stock!
$libinclude xldump CO2_capture_isi_stock.l        %outputfile% CO2_capture_isi_stock!
$libinclude xldump CO2_capture_chemical_stock.l   %outputfile% CO2_capture_chemical_stock!
$libinclude xldump CO2_capture_nh3_stock.l        %outputfile% CO2_capture_nh3_stock!
$libinclude xldump CO2_capture_meoh_stock.l       %outputfile% CO2_capture_meoh_stock!
$libinclude xldump CO2_capture_ref_stock.l        %outputfile% CO2_capture_ref_stock!
$libinclude xldump CO2_capture_liquid_stock.l     %outputfile% CO2_capture_liquid_stock!
$libinclude xldump CO2_capture_ngas_stock.l       %outputfile% CO2_capture_ngas_stock!
$libinclude xldump CO2_capture_olefin_stock.l     %outputfile% CO2_capture_olefin_stock!
$libinclude xldump CO2_capture_glycol_stock.l     %outputfile% CO2_capture_glycol_stock!
$libinclude xldump CO2_capture_total_stock.l      %outputfile% CO2_capture_total_stock!

$libinclude xldump CO2_inject_stock.l          %outputfile% CO2_inject_stock!
$libinclude xldump num_total_well_new.l        %outputfile% num_total_well_new

$libinclude xldump num_pipes_total.l              %outputfile% num_pipes_total!
$libinclude xldump num_pipe_stock.l               %outputfile% num_pipe_stock!
$libinclude xldump CO2_flow_stock.l               %outputfile% CO2_flow_stock!
$libinclude xldump Total_CO2_Pipeline_Length.l    %outputfile% Total_CO2_Pipeline_Length!

$libinclude xldump Cost_Capture.l    %outputfile% Cost_Capture!
$libinclude xldump Cost_Storage.l    %outputfile% Cost_Storage!
$libinclude xldump Cost_Transport.l    %outputfile% Cost_Transport!
$libinclude xldump Revenue_EOR.l    %outputfile% Revenue_EOR!

$libinclude xldump objective.l    %outputfile% objective!
