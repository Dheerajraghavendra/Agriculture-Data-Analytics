fileID = fopen('advices1.txt');
loc = textscan(fileID,'%s %s %s %*[^\n]');
fclose(fileID);
c = [loc{2} loc{1}]
FID = fopen('crops2.txt','w');
formatspec = '%s %s \r\n';
[a,b] = size(c);
for i = 1:a
    fprintf(FID,formatspec,c{i,:});
end
fclose(FID);
type crops2.txt