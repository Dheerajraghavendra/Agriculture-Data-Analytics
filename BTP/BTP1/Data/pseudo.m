fid = fopen('advices1.txt');
a2 = textscan(fid,'%s','delimiter','\n','whitespace', '');
a2=a2{1};
[x,y]=size(a2);
for i = 1:x
    date{i} = a2{i}(1:10);
    farm{i} = a2{i}(12:56);  
    crop{i} = strread(a2{i}(58:end),'%s %*[^\n]');
    obs{i} = a2{i}(61:end);
end
fclose(fid);
flag=zeros(x,1);
k=1;
for i=1:x
    if(flag(i)==0)
        flag(i)=1;
        count(k)=1;
        newcrop{k}=crop{i};
        for j=i+1:x
            if(strcmp(crop{j},crop{i}))
                count(k)=count(k)+1;
                flag(j)=1;
            end
        end
        k=k+1;
    end
end
fid2= fopen('cropwise.txt','w');
for i=1:k-1
    fprintf(fid2,'%s',newcrop{i}{1});
    fprintf(fid2,' %d',count(i));
    fprintf(fid2,'\r\n');
end
fclose(fid2);