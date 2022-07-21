def visualize_data_loader(data_loader):
  for batch in data_loader:
    for data in batch:
      print(data)
    break
      