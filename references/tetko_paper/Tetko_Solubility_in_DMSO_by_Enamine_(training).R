
reorder = function(m) m[order(rownames(m)),]

models$cv$rmse = matrix(models$cv$rmse, ncol = length(models$cv$methodNames), nrow = length(models$cv$descriptorNames), dimnames = list(models$cv$descriptorNames, models$cv$methodNames))
models$cv$rmse = reorder(models$cv$rmse)

models$cv$r2 = matrix(models$cv$r2, ncol = length(models$cv$methodNames), nrow = length(models$cv$descriptorNames), dimnames = list(models$cv$descriptorNames, models$cv$methodNames))
models$cv$r2 = reorder(models$cv$r2)

models$cv$q2 = matrix(models$cv$q2, ncol = length(models$cv$methodNames), nrow = length(models$cv$descriptorNames), dimnames = list(models$cv$descriptorNames, models$cv$methodNames))
models$cv$q2 = reorder(models$cv$q2)

models$cv$mae = matrix(models$cv$mae, ncol = length(models$cv$methodNames), nrow = length(models$cv$descriptorNames), dimnames = list(models$cv$descriptorNames, models$cv$methodNames))
models$cv$mae = reorder(models$cv$mae)
models = list()
models$Bagging = list()
models$Bagging$auc = c(NA,0.8,NA,NA,NA,NA,NA,NA,NA,0.79,NA,NA,0.74,NA,0.79,NA,0.78,NA,0.8,NA,0.8,NA,0.79,NA,NA,0.78,NA,NA,NA,0.81,NA,0.8,0.81,0.8,NA,0.81,NA,0.81,0.79,NA,NA,0.78,NULL)
models$Bagging$methodNames = c("WEKA-RF  (CHEMAXON)","WEKA-J48  (CHEMAXON)","ASNN   (CHEMAXON)",NULL)
models$Bagging$rmse = c(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NULL)
models$Bagging$mae = c(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NULL)
models$Bagging$r2 = c(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NULL)
models$Bagging$q2 = c(NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NULL)
models$Bagging$accuracy = c(NA,74,NA,NA,NA,NA,NA,NA,NA,76,NA,NA,70,NA,81,NA,82,NA,78,NA,82,NA,81,NA,NA,80,NA,NA,NA,76,NA,76,76,76,NA,77,NA,77,74,NA,NA,73,NULL)
models$Bagging$descriptorNames = c("ALogPS, OEstate","ALogPS, OEstate","Adriana (3D by Adriana) 3D:corina","Adriana (3D by Adriana) 3D:corina","CDK (cons,topol,geom,elect,hybr) 3D:corina","ChemaxonDescriptors (pH 0 - 14:1) 3D:corina","Dragon6 (3D blocks: (only) 1-29) 3D:corina","Dragon6 (3D blocks: (only) 1-29) 3D:corina","Fragmentor (length: 2-4)","Fragmentor (length: 2-4)","GSFrag (F + L)","GSFrag (F + L)","InductiveDescriptors 3D:corina","Mera, Mersy 3D:corina",NULL)
models$Bagging$balancedAccuracy = c(NA,73,NA,NA,NA,NA,NA,NA,NA,72,NA,NA,67,NA,71,NA,70,NA,72,NA,71,NA,71,NA,NA,69,NA,NA,NA,74,NA,73,74,73,NA,73,NA,74,72,NA,NA,71,NULL)

reorder = function(m) m[order(rownames(m)),]

models$Bagging$rmse = matrix(models$Bagging$rmse, ncol = length(models$Bagging$methodNames), nrow = length(models$Bagging$descriptorNames), dimnames = list(models$Bagging$descriptorNames, models$Bagging$methodNames))
models$Bagging$rmse = reorder(models$Bagging$rmse)

models$Bagging$r2 = matrix(models$Bagging$r2, ncol = length(models$Bagging$methodNames), nrow = length(models$Bagging$descriptorNames), dimnames = list(models$Bagging$descriptorNames, models$Bagging$methodNames))
models$Bagging$r2 = reorder(models$Bagging$r2)

models$Bagging$q2 = matrix(models$Bagging$q2, ncol = length(models$Bagging$methodNames), nrow = length(models$Bagging$descriptorNames), dimnames = list(models$Bagging$descriptorNames, models$Bagging$methodNames))
models$Bagging$q2 = reorder(models$Bagging$q2)

models$Bagging$mae = matrix(models$Bagging$mae, ncol = length(models$Bagging$methodNames), nrow = length(models$Bagging$descriptorNames), dimnames = list(models$Bagging$descriptorNames, models$Bagging$methodNames))
models$Bagging$mae = reorder(models$Bagging$mae)
models$no-validation = list()
models$no-validation$methodNames = c("Consensus:AVERAGE (CHEMAXON)",NULL)
models$no-validation$accuracy = c(79,NULL)
models$no-validation$descriptorNames = c("Table 3 - Consensus",NULL)
models$no-validation$auc = c(0.82,NULL)
models$no-validation$balancedAccuracy = c(74,NULL)

reorder = function(m) m[order(rownames(m)),]

models$no-validation$rmse = matrix(models$no-validation$rmse, ncol = length(models$no-validation$methodNames), nrow = length(models$no-validation$descriptorNames), dimnames = list(models$no-validation$descriptorNames, models$no-validation$methodNames))
models$no-validation$rmse = reorder(models$no-validation$rmse)

models$no-validation$r2 = matrix(models$no-validation$r2, ncol = length(models$no-validation$methodNames), nrow = length(models$no-validation$descriptorNames), dimnames = list(models$no-validation$descriptorNames, models$no-validation$methodNames))
models$no-validation$r2 = reorder(models$no-validation$r2)

models$no-validation$q2 = matrix(models$no-validation$q2, ncol = length(models$no-validation$methodNames), nrow = length(models$no-validation$descriptorNames), dimnames = list(models$no-validation$descriptorNames, models$no-validation$methodNames))
models$no-validation$q2 = reorder(models$no-validation$q2)

models$no-validation$mae = matrix(models$no-validation$mae, ncol = length(models$no-validation$methodNames), nrow = length(models$no-validation$descriptorNames), dimnames = list(models$no-validation$descriptorNames, models$no-validation$methodNames))
models$no-validation$mae = reorder(models$no-validation$mae)
