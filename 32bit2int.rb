require 'csv'
i = 0
#Transform cat variables to 16bit
CSV.open("train50k_16bit.csv", "wb") do |csv|
  CSV.foreach("train50k.csv") do |row|


    if i == 0
      i = 1
      next
    end

    (15..40).each do |j|
      if row[j]
        row[j] = row[j].to_i(16)
      end
    end
    csv << row

  end
end
