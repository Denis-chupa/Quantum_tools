
#
# Compile and run MOSEK examples
#
examples.jar: TrafficNetworkModel.java alan.java baker.java breaksolver.java callback.java cqo1.java ceo1.java pow1.java djc1.java diet.java duality.java elastic.java facility_location.java lo1.java gp1.java logistic.java lownerjohn_ellipsoid.java lpt.java milo1.java mioinitsol.java mico1.java nearestcorr.java opt_server_sync.java parameters.java parallel.java pinfeas.java portfolio_1_basic.java portfolio_2_frontier.java portfolio_3_impact.java portfolio_4_transcost.java portfolio_5_card.java portfolio_6_factor.java primal_svm.java reoptimization.java response.java qcqp_sdo_relaxation.java sdo1.java sdo2.java sdo3.java sospoly.java sudoku.java total_variation.java tsp.java
	javac -d . -classpath ..\..\..\platform\win64x86\bin\mosek.jar TrafficNetworkModel.java alan.java baker.java breaksolver.java callback.java cqo1.java ceo1.java pow1.java djc1.java diet.java duality.java elastic.java facility_location.java lo1.java gp1.java logistic.java lownerjohn_ellipsoid.java lpt.java milo1.java mioinitsol.java mico1.java nearestcorr.java opt_server_sync.java parameters.java parallel.java pinfeas.java portfolio_1_basic.java portfolio_2_frontier.java portfolio_3_impact.java portfolio_4_transcost.java portfolio_5_card.java portfolio_6_factor.java primal_svm.java reoptimization.java response.java qcqp_sdo_relaxation.java sdo1.java sdo2.java sdo3.java sospoly.java sudoku.java total_variation.java tsp.java
	jar cf examples.jar com

clean:
	del examples.jar
	rmdir /s /q com

test: examples.jar
	set PATH=..\..\..\platform\win64x86\bin;%%PATH%%
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.TrafficNetworkModel
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.alan
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.baker
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.breaksolver
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.callback
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.cqo1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.ceo1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.pow1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.djc1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.diet
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.duality
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.elastic
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.facility_location
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.lo1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.gp1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.logistic
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.lownerjohn_ellipsoid
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.lpt
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.milo1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.mioinitsol
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.mico1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.nearestcorr
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.opt_server_sync
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.parameters
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.parallel
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.pinfeas
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.portfolio_1_basic
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.portfolio_2_frontier
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.portfolio_3_impact
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.portfolio_4_transcost
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.portfolio_5_card
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.portfolio_6_factor
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.primal_svm
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.reoptimization
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.response
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.qcqp_sdo_relaxation
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.sdo1
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.sdo2
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.sdo3
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.sospoly
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.sudoku
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.total_variation
	java -cp examples.jar;..\..\..\platform\win64x86\bin\mosek.jar com.mosek.fusion.examples.tsp
