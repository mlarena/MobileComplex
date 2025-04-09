import geocoder

def get_current_location():
    g = geocoder.ip('me')
    if g.latlng:
        return g.latlng
    else:
        return None

current_location = get_current_location()
if current_location:
    print(f"Текущие координаты: {current_location}")
else:
    print("Не удалось определить текущие координаты.")
