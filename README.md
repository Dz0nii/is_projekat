Najbitniji deo, rzlog iz kog mi prva iteracija podataka nije bila dobra je sto nisam podesio CRS, sto je izgleda projekcija. Znaci mora da se postavi 
Project->Properties->CRS->EPSG:4326

Posle toga treba sacuvati sliku satelita
Project->Import/export->Map to image
Koristiti 300dpi i SACUVATI KAO TIFF!

Onda mogu iskljuciti gugl satelit i uvesti samo ovaj upravo sacuvani tiff fajl preko layer->add layer->raaster layer

e sad se preko quickOSM-a dodaju parkovi
i onda Raster->Conversion->Vector to raster
Koristiti extent ono kao layer i koristit ovaj sad napravljeni

tjt ja mislim sto se tice qgisa i onda ide pajton

Imamo skriptu tile_city gde samo treba izmeniti putanje do satelita i do maske i ime grada
Skripta za proveru maski jer se paranoja isplati.