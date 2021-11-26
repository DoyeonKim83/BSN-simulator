# BSN-simulator

Binary Stochastic Neuron 생성을 위한 시뮬레이터이다. -5.0 ~ 5.0의 input number를 이용해 BSN을 생성하고 그 결과를 총 3개의 그래프로 보여준다. BSN의 확률 값은 sigmoid function을 통해 구해지며 이 확률 값과 uniform distribution에서 random 하게 얻은 값을 이용해 BSN이 구해진다. 각 output graph는 각각의 retention time을 가지고 있으며, 시뮬레이터의 성능을 위해 Sampling이 적용된다. Sampling Time은 일반적으로 retention time의 2~3배로 사용자가 설정하며, Timer 기능을 통해 일정 시간마다 Sampling Queue에 BSN이 삽입되는 로직이다. Sampling Queue가 Full 상태이고 Sampling Number 주기 또한 Full 이면 MEAN graph와 Bit Configuration Graph가 갱신된다.

<br>
<br>

![5 graph_000_3_100](https://user-images.githubusercontent.com/47073695/143517027-ebbaf1e3-63ec-4347-b5b0-72329e17d499.PNG)

<br>
<br>

[Graph Info]
- BSN output : 각 input number를 통해 생성된 BSN을 연속적으로 보여줌, Always Updating
- BSN Mean : BSN output의 평균 (Sampling 적용)
- Bit Configuration : 생성된 각각의 BSN들을 이어 붙여서 만든 2진수 정수의 분포도 (Sampling 적용)
<br>
[Main Function]<br><br>
1. 비트 개수 입력 -> 동적으로 parameter 생성<br>
2. Parameter SAVE as File -> LOAD at Graph Simulator -> Graph 동적 생성<br>
3. The 3 graph for each function : BSN output, BSN Mean, Bit Configuration<br>
4. Retention time for each input and Using sampling<br>
5. Random input number update<br>
6. Again, parameter SAVE as File -> RELOAD at Graph Simulator<br>
7. Load parameter value at Parameter Simulator
<br><br><br>

- Parameter 입력을 위한 simulator와 BSN Graph 출력을 위한 simulator가 분리되어 있는 프로그램이다. Parameter Simulator에서 입력한 Parameter을 csv 파일로 저장하고 이를 Graph Simulator에서 불러오는 방식으로 실행된다. 
- Parameter Simulator에서는 parameter 입력은 물론이고, parameter value 저장 및 불러오기와 Random input number를 설정하는 기능을 포함하고 있다.
- Graph Simulator는 file에서 불러온 parameter value를 통해 3개의 Graph를 동적 생성한다.

<br>

[File Info]
- BSN simulator_param : Parameter 입력을 위한 simulator
- BSN simulator_graph : Graph 출력을 위한 simulator
- BSN simulator_version 5 : Parameter simulator와 Graph simulator를 하나로 합친 버전
