library(airGR)
library(microbenchmark)
data(L0123001)

# 设置数据长度限制
data_length <- 500  # 可以根据需要修改为其他值

# 创建输入模型
InputsModel <- CreateInputsModel(FUN_MOD = RunModel_GR4J, DatesR = BasinObs$DatesR,
                                 Precip = BasinObs$P, PotEvap = BasinObs$E)

# 控制数据长度
Ind_Run <- seq(which(format(BasinObs$DatesR, format = "%Y-%m-%d") == "1990-01-01"), 
               which(format(BasinObs$DatesR, format = "%Y-%m-%d") == "1990-01-01") + data_length - 1)

# 设置初始状态
IniResLevels <- as.double(c(0.3, 0.5, NA, NA))
RunOptions <- CreateRunOptions(FUN_MOD = RunModel_GR4J,
                               InputsModel = InputsModel, IndPeriod_Run = Ind_Run,
                               IniStates = NULL, IniResLevels = IniResLevels, IndPeriod_WarmUp = NULL)

# 设置模型参数
Param <- c(320.11, 2.42, 69.63, 1.39)

# 运行模型并计时
start_time <- Sys.time()
OutputsModel <- RunModel_GR4J(InputsModel = InputsModel, RunOptions = RunOptions, Param = Param)
end_time <- Sys.time()
run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
cat(sprintf("模型运行时间: %.3f 秒\n", run_time))

# 使用microbenchmark进行多次运行测试
times <- microbenchmark(
  RunModel_GR4J(InputsModel = InputsModel, RunOptions = RunOptions, Param = Param),
  times = 1000
)
cat(sprintf("1000次运行的平均时间: %.3f 毫秒\n", mean(times$time) / 1e6))

# 绘制结果
plot(OutputsModel, Qobs = BasinObs$Qmm[Ind_Run])

# 创建输出数据框
df <- data.frame(
  time = OutputsModel$DatesR,
  pet = OutputsModel$PotEvap,
  prec = OutputsModel$Precip,
  prob = OutputsModel$Prod,
  pn = OutputsModel$Pn,
  ps = OutputsModel$Ps,
  evap = OutputsModel$AE,
  perc = OutputsModel$Perc,
  pr = OutputsModel$PR,
  q9 = OutputsModel$Q9,
  q1 = OutputsModel$Q1,
  rout = OutputsModel$Rout,
  exch = OutputsModel$Exch,
  aexch1 = OutputsModel$AExch1,
  aexch2 = OutputsModel$AExch2,
  qr = OutputsModel$QR,
  qd = OutputsModel$QD,
  qsim = OutputsModel$Qsim,
  qobs = BasinObs$Qmm[Ind_Run]
)

# 输出数据长度信息
cat(sprintf("处理的数据长度: %d\n", nrow(df)))
