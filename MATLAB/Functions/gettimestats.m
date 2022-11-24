function [days,hours,minutes,seconds] = gettimestats(total)
    days = floor(total/(24*60*60));
    
    total = total-24*60*60*days;
    hours = floor(total/(60*60));
    
    total = total-60*60*hours;
    minutes = floor(total/60);

    total = total-60*minutes;
    seconds = round(total);

    days = time2str(days);
    hours = time2str(hours);
    minutes = time2str(minutes);
    seconds = time2str(seconds);

    function str = time2str(time)
        assert(time>=0,"how is time negative\n");
        if time < 10
            str = append('0',num2str(time));
        else
            str = num2str(time);
        end
    end
end