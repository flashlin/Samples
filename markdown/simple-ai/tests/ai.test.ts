import { describe, expect, test } from "@jest/globals";
import { Neuron, Neuron2, convertIdToNumbers, program, train } from '@/ai'; 

test('ai1', () => {
  const result = program(1);

  expect(result).toBe(1);
});


test('ai2', () => {
  const ids = ["E1735036210","R278834622","B237836243","D244273034","O2019822310","Y271964122","C239484837","P263820767","G231218906","B190165729","K124550463","E184776282","X286428383","B220160145","C289788862","P164150628","C1212528110","C181628116","R279811516","Q268712083","S299458141","F163801352","P255264157","K198937994","X209957734","F120852882","H220807345","X275488186","D185707646","A275091372","K155181836","U289953935","B195715372","K1531103210","N1678480810","E1675126410","H161935964","B2530905510","S170272496","A131231287","G237413996","K234703147","U158095979","C222408087","H225286891","F1460663410","O295989885","V263650743","K243702176","O2043132010",
  "C2292984910","Q199873082","J278402142","X156826753","T192211715","B123152296","W202752564","U1085723510","O276118008","O132754335","V1727398810","Y1214883810","K139273697","Y2076807410","M113295185","M266796275","E111279285","V293882619","X1027610510","F155308628","C100564931","G253795493","Y236427104","Y119995772","A160264512","V177394863","F210405971","Z152522015","K235004858","F191983949","V249075999","P138709539","C115218571","P291404504","M258319335","E226445524","P179509535","I268451931","E122597838","W162377585","F290077177","D222169853","L283131286","P204428732","I170319795","Z154548446","L288937297","N124928669","U241636926","G162445388"];
  const inputs = ids.map(x => convertIdToNumbers(x).slice(0, 9));
  const targets = ids.map(x =>convertIdToNumbers(x).slice(9)[0]);

  const neuron = new Neuron2(10);
  train(inputs, targets, neuron);

  const idStr = "U17744709"; //1
  const idNumbers = convertIdToNumbers(idStr);
  const output = neuron.forward(idNumbers);
  console.log(output);
});