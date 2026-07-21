from datetime import datetime, timezone, timedelta

ist = timezone(timedelta(hours=5, minutes=30))
print(datetime.fromtimestamp(1774955979.717341, ist))

#checks the time in ist