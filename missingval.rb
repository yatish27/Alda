require 'csv'

missing_values = %W|1
3
7
5
2907
39
3
8
1
1
1
0
5
98275684
950618017
3492987491
3247169921
633879704
2114768079
1062127239
185940084
2805916944
990438539
3299735832
3753576955
2245779768
131152527
913448412
2223606570
3854202482
1525511222
568184265
1480633834
1360682
2905629419
851920782
395552808
3904386055
1238795398|

j = 0
CSV.open("train50k_16bit_missing_vals.csv", "wb") do |csv|
  CSV.foreach("train50k_16bit.csv") do |row|

    if j == 0
      j = 1
      next
    end

    (2..40).each do |i|
      if !row[i]
        row[i] = missing_values[i-2]
      end
    end
    csv << row
  end
end
