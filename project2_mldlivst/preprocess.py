import general_preprocess as gPre

def run(df):
  res = gPre.run(df)
  print ("p1 preprocess")
  # encoderF4 = gPre.encodeCtgs(df, "f4")
  # encoderF5 = gPre.encodeCtgs(df, "f5")
  # res["ctg_f4"] = encoderF4
  # res["ctg_f5"] = encoderF5
  return res