ALTER TABLE store
  ADD COLUMN gis character varying(255);
update store set gis = '20.983764,-89.617983' where store_city = 'Merida';
update store set gis = '20.674212,-103.349682' where store_city = 'Guadalajara';
update store set gis = '22.760511,-102.572707' where store_city = 'Camacho';
update store set gis = '16.863794,-99.881614' where store_city = 'Acapulco';
update store set gis = '18.851848,-97.103498' where store_city = 'Orizaba';
update store set gis = '23.195614,-102.885209' where store_city = 'Hidalgo';
update store set gis = '19.491447,-99.180043' where store_city = 'San Andres';
update store set gis = '19.445255,-99.147198' where store_city = 'Mexico City';
update store set gis = '37.765278, -122.240556' where store_city = 'Alameda';
update store set gis = '34.052222, -118.242778' where store_city = 'Los Angeles';
update store set gis = '32.715278, -117.156389' where store_city = 'San Diego';
update store set gis = '37.775000, -122.418333' where store_city = 'San Francisco';
update store set gis = '34.073611, -118.399444' where store_city = 'Beverly Hills';
update store set gis = '45.523611, -122.675000' where store_city = 'Portland';
update store set gis = '44.943056, -123.033889' where store_city = 'Salem';
update store set gis = '48.759722,-122.486944' where store_city = 'Bellingham';
update store set gis = '47.567500,-122.631389' where store_city = 'Bremerton';
update store set gis = '47.606389, -122.330833' where store_city = 'Seattle';
update store set gis = '47.658889,-117.425000' where store_city = 'Spokane';
update store set gis = '47.253056,-122.443056' where store_city = 'Tacoma';
update store set gis = '46.064722,-118.341944' where store_city = 'Walla Walla';
update store set gis = '46.602222,-120.504722' where store_city = 'Yakima';
update store set gis = '48.428127,-123.360234' where store_city = 'Victoria';
update store set gis = '49.259041,-123.114614' where store_city = 'Vancouver';