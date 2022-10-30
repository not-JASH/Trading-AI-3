function display_number(string,number)
    a = num2str(number);
    b = '';

    while length(a) > 3
        b = append(',',a(end-2:end),b);
        a(end-2:end) = [];
    end
    b = append(a,b);
    fprintf("%s : %s\n",string,b);
end