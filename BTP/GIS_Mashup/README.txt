h1. Description

A mashup using the Google maps API to show the location of the Foodmart stores and show 
performance reports for each of them when clicked on the map.
The GMap receives the information for locating the markers from the Foodmart DB.

h1. Install

You will find 3 files and 1 folder:
1 - foodmart-gis.sql => Import this into your foodmart db. It will add a new column for the Geo Location information to the store table and fill in the city locations.
2 - gmaps_gis.jsp => drop this JSP into your /webapps/jasperserver-pro folder
3 - images (folder) => drop this JSP into your /webapps/jasperserver-pro folder
4 - GIS-v0.5.zip => import this into your JRS repository
5 - Go to  JRS 4.7 and execute the dasboard in /Public/GIS
The sample assumes that JasperServer is installed in /jrs-pro change that value in gmaps_gis.jsp
to match your instalation.

h1. Changing the Reports 

I reference some of the reports and dashboards form the Mexico Sample so if you don't have
that you may need it to see all the reports or you can change the paths on the JSP to point
to the reports or the original Google maps sample.

The markers in the map come from the database using the RESTv2 API. A dummy report "StoreGIS" 
in /public/GIS has a single select query input control that returns me the "store city" as a
value and the GeoLocation (lat, long) as a label. All the markers returned from that iC will be 
added to the map.
The value is used as a parameter to pass to the reportUnit loaded when the user clicks on the
markers. To change this modify the replaceURL(param) function in gmaps_gis.jsp.
Also in the file check that the variable JRSPath (~line 97) matches your install.



