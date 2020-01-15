fid = fopen('SampleAssessmentbaeb56e.csv');%opening the file
Header = textscan(fid,'%s %s %s %s %s %s %s %s %s %s %s',1, 'delimiter',',','whitespace', '');%header
data = textscan(fid, '%s %s %s %s %s %s %s %s %s %s %s','delimiter',',','whitespace', '');%data
fclose(fid);
l=length(data{1});
for i 