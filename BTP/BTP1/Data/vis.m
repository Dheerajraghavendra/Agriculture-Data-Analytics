fid = fopen('crops2.txt');
t = textscan(fid,'%s %s \r\n');
a = t{1};
b = t{2};
fclose(fid);
[x,y]=size(a);
h = zeros(x,1);
%p = cell(x,1);
itr=0;
s = ' ';
for i=1:x
    itr2=1;
    if h(i)==0
        itr = itr+1;
        s = a{i}(1:34);
        p{itr}{itr2}= s;
        itr2 = itr2+1;
        for j = i:x
            if (strcmp(s,a{j}(1:34)))&&(h(j)==0)
                h(j)=1;
                p{itr}{itr2}= b{j};
                itr2 = itr2+1;
            end
        end
    end
end
