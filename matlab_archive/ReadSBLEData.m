% Script to read SBLE data, compute statistics reported in the paper, and
% generate plots.
%
% User is prompted to first select the folder that contains the files. Then,
% the user is prompted to select the data file and the notification file.
% 
% If DISPLAY_DATA is true, it shows the pdf of the joint RSSI from both
% beacons in same bus, for the user in front, middle, or read of bus
%
% If additionally DISPLAY_PLOT is true, it shows routes and RSSI plots for
% each trip (red:pretrips; blue:pretrips)
%
% Note: In this code, "pretrips" indicates "Detecting beacons" state; "trip"
% indicates "Traveling in bus" state.
%
% Length of routes are computed only when distance function is available in
% Matlab (since R2019b)
%
% Reference: J. Lam, R. Manduchi, "S-BLE: A Participatory BLE Sensory Data
% Set Recorded from Real-World Bus Travel Events", submitted to TRB 2024.

%%
DISPLAY_DATA = true; 
DISPLAY_PLOT = false; % If true, DISPLAY_DATA must also be true
DISPLAY_SEATS = true;

%% Read data files (data + notification)
% if 0
%     dirname = '/Users/manduchi/UCSC/Projects/NSF S&CC 21/Data';
%     currdir = pwd;
%     cd(dirname);
%     fname_d = fullfile(dirname,uigetfile('SBLE_data*.csv'));
%     SBLE_d = readtable(fname_d);
%     fname_n = fullfile(dirname,uigetfile('SBLE_notification*.csv'));
%     SBLE_n = readtable(fname_n);
%     cd(currdir);
% else
%     SBLE_d = readtable('SBLE_data November 23.csv');
%     SBLE_n = readtable('SBLE_notification November 23.csv');
% end  
%%
currdir = pwd;
theDir = '/Users/Jake/Computer Science/SBLE_project/data';
% fn_data = dir('/Users/manduchi/UCSC/Projects/NSF S&CC 21/Data/*_data*');
% fn_not = dir('/Users/manduchi/UCSC/Projects/NSF S&CC 21/Data/*notification*');

fn_data = dir('/Users/Jake/Computer Science/SBLE_project/data/*_data*');
fn_not = dir('/Users/Jake/Computer Science/SBLE_project/data/*notification*');

SBLE_d = []; SBLE_n = [];
for i = 1:length(fn_data)
    SBLE_d = vertcat(SBLE_d,readtable(fullfile(fn_data(i).folder,fn_data(i).name)));
end

SBLE_d = sortrows(SBLE_d,["timestamp", "rssi"],"ascend");

for i = 1:length(fn_not)
    SBLE_n = vertcat(SBLE_n,readtable(fullfile(fn_not(i).folder,fn_not(i).name)));
end

SBLE_n = sortrows(SBLE_n,"timestamp","ascend")


%%
stopsGTFS = readtable('/Users/Jake/Computer Science/SBLE_project/data/stops.csv');
%%
if DISPLAY_PLOT
    figure(1),clf
    figure(2),clf
end

clear NTrips4n  NSpur4n theTimes theTimesWithSpur theTimesSpur theTimesOut theTimes5 theTimesOut5 totTimes totDistn totTrips theTimesWithSpur5 theTimesSpur5 nOnly1B whichM whichM0
clear RSSIdiff  SL noMinor1 noMinor2 theTS isEmptyMinor theMM lengthEitherMinor n_replied_no theGaps LL1 LL2 LL1pt LL2pt nClips LL LLpt allM totS totSspur nClips
clear tDur totNS totNS1 totNS2 trips pretrips didNotMarkExit LLM totNS_goodbus totNS1and2 totNS1or2_5

totDist = [];

% These are the majors for which there is a missing minor (2 for 2,8 and 1 for 21)
badMM = [2 8 21];

% Find unique names
names = unique(SBLE_d.username);

MIN_RSSI = -100;
% let's keep RSSI=0 but set it to a very low value
SBLE_d.rssi(SBLE_d.rssi==0) = MIN_RSSI;

% Search for each user
for n = 1 :length(names)
    % notification file
    % notifications for n-th user
    J = ismember(SBLE_n.username, names(n));
    JJ = 1:length(J); JJ = JJ(J);
    
    % find all of the completed sequences
    i = 1;
    tripInd = 1; pretripInd = 1; 

    while i < length(JJ)
        % find the new collecting_data. If a sequence of TRUE, find the
        % last one in the sequence.
        % Note: a collecting_data TRUE that is not followed by
        % sitting_on_bus indicates that the user did not enter a bus, or
        % that a user did not confirm that they are ina bus.
        % Multiple collecting_data TRUE are only possible if the user
        % closed the app or it crashed
        while isequal(SBLE_n.message_type{JJ(i)}, 'collecting_data') 
            i = i+1;
            if i > length(JJ)
                break;
            end
        end
        if i > length(JJ)
            % We arrived at the end of the notification for the n-th user
            break;
        end
        
        % We are at (the end of) a sequence of collecting_data TRUE which should be followed by
        % a sitting_on_bus
        % --> For now let's only consieder sitting_on_bus TRUE. We should
        % also see if there is any that is FALSE - that should be
        % interesting!        

        % mark the initial timestamp
        pretrips(n).t{tripInd}.init = SBLE_n.timestamp(JJ(i-1));
        if ~isequal(SBLE_n.message_type{JJ(i)}, 'sitting_on_bus')
            % RM 11/17 This may happen if this is the last sequence with
            % this user and they didn't take the bus
            error('There should be a sitting_on_bus here!');
        end
        isTrip = false;
        if (isequal(SBLE_n.message{JJ(i)}, 'TRUE') || isequal(SBLE_n.message{JJ(i)}, 'true')) 
            isTrip = true;
            trips(n).t{tripInd}.init = SBLE_n.timestamp(JJ(i));
            trips(n).t{tripInd}.seat_location = '';
            
            % Note that there is a 1 sample overlap between pretrips and trip
           if ~(isequal(SBLE_n.message_type{JJ(i-1)},'collecting_data') ...
                &&  (isequal(SBLE_n.message{JJ(i-1)}, 'TRUE') || isequal(SBLE_n.message{JJ(i-1)}, 'true')))
            % This last test is necessary because in one case the
            % collecting_data was not sent, so it was missed and we don't
            % know when the pretrips started. We need to clear the pretrips.
            % Let's just make it of 1 sample, at the beginning of the trip
            
                disp('sitting_on_bus without preceding collecting_data=TRUE')
                pretrips(n).t{tripInd}.init = SBLE_n.timestamp(JJ(i));
           end
           pretrips(n).t{tripInd}.end = SBLE_n.timestamp(JJ(i));
         else
            % False alarm - let's just close it for now.
            disp('sitting_on_bus FALSE - do something!')
            isTrip = false;
        end      
         % now find the end time. Before that, there should be a
         % seat_location
         % Note that  the user may not have tapped the "have you left the bus"
        % Either case, the next 'collecting_data' event is what we are
        % looking for
         
        n_replied_no(n).t{tripInd} = 0;
        
        if isTrip %RM 9-27-24
         while ~(isequal(SBLE_n.message_type{JJ(i)}, 'collecting_data') && ~isequal(SBLE_n.message{JJ(i)}, 'answered no'))
 
             % RM 7/27/24 had to put this but not clear why
             SL(n).t{tripInd} = [];
             
             if isequal(SBLE_n.message_type{JJ(i)}, 'seat_location') 
                 % Note that this may be missing if the user never replied to the
                 % seat_location notification. In this case, the 
                 % notification would be repeated until the user closes the
                 % app.
                 trips(n).t{tripInd}.seat_location = SBLE_n.message{JJ(i)};
                 switch trips(n).t{tripInd}.seat_location
                     case 'front'
                         SL(n).t{tripInd} = 1;
                     case 'middle'
                         SL(n).t{tripInd} = 2;
                     case 'back'
                         SL(n).t{tripInd} = 3;
                 end

             elseif (isequal(SBLE_n.message_type{JJ(i)}, 'collecting_data') && isequal(SBLE_n.message{JJ(i)}, 'answered no'))
                 % Need to count these. The system thought the user was off
                 % the bus and the user denied it
                 n_replied_no(n).t{tripInd} = n_replied_no(n).t{tripInd} + 1;
             end
             i = i+1;
             if i > length(JJ)
                break;
             end
         end
        

            % OK now it is a new collecting_data. In theory, it should be
            % FALSE, unless the app crashed or the user restarted. Either way,
            % it is the end of a trip

            % In theory, the user should have said 'false'. Let's see if this
            % is not the case

            didNotMarkExit(n).t{tripInd} = false;
            if (i > length(JJ)) || ~( isequal(SBLE_n.message{JJ(i)}, 'FALSE') || isequal(SBLE_n.message{JJ(i)}, 'false'))
                didNotMarkExit(n).t{tripInd} = true;
            end


  %          if isTrip
                % Mark the end of a trip
                % RM 11/17: at this JJ(i), a collecting_data is received,
                % unless we are at the end of collection

                if i > length(JJ)
                    trips(n).t{tripInd}.end = SBLE_n.timestamp(JJ(i-1));
                else    
                    % Note that we are ending the trip at a data_collection. If data_collection is FALSE, that's
                    % ok - it confirms that the user is out of the bus. If it
                    % is TRUE, then the app got closed/crashed, and we are
                    % starting to collect data. Note that it is possible that
                    % data from other beacons has been collected in this period
                    % if the app did not get closed.
                    trips(n).t{tripInd}.end = SBLE_n.timestamp(JJ(i));
                end           

                if isempty(trips(n).t{tripInd}.end)
                    disp('Something''s wrong')
                end
                tripInd = tripInd + 1;
            end
            if i > length(JJ)
                break
            end      
            i = i+1;
        end

        % OK now we have the sequence of trips and pretrips start and ending
        % times. Let's find the associated data sequences.

        theTimes(n) = 0; theTimesSpur(n) = 0; theTimesOut(n) = 0; theTimesWithSpur(n) = 0; theTimes5(n) = 0; theTimesOut5(n) = 0; totDistn(n) = 0; totTrips(n) = 0; theTimesWithSpur5(n) = 0; theTimesSpur5(n) = 0; 

        % data file
        % Indices for user n 
        I =  ismember(SBLE_d.username, names(n));

        % Indices with minor 1 or 2
        II1 = (SBLE_d.minor == 1); 
        II2 = (SBLE_d.minor == 2);

        % Indices with speed >= 5 m/s
        Is5 = SBLE_d.speed >= 5;    

        IItotn = false(size(I)); 

        % Note that data is repeated for minor 1 and 2 (when data from both
        % minors is received). This includes timestamps. 

        % Unique timestamps for user n
        theTimesOut(n) = length(unique(SBLE_d.timestamp(I)));


        if length(trips) == n % check that there is data

        % Loop through all completed sequences for user n
        %trpIndInd = 1;
        for tripInd = 1:length(trips(n).t)

            % logical Indices for user n for the trip of index tripInd      
            II = false(size(I));

            theII = []; theIIind = 1; theIIpt = [];
            auxTd = inf(size(SBLE_d.timestamp));
            auxTd(I) = SBLE_d.timestamp(I); % timestamps for n-th user

            % Find the logical indices of timestamps that are within the tripInd-th
            % trip for user n

            % RM 11/30 only consider samples in data that are older than in
            % notification
    %            aux = abs(auxTd - trips(n).t{tripInd}.init);
            aux = auxTd - trips(n).t{tripInd}.init; aux(aux <= 0) = inf;
            [~,ind1] = min(aux);

    %            aux = abs(auxTd - trips(n).t{tripInd}.end);
%            aux = auxTd - trips(n).t{tripInd}.end; aux(aux <= 0) = inf;
%            [~,ind2] = min(aux); % note that there may be more than 1 
            % RM 7/28/24
            auxTd = -inf(size(SBLE_d.timestamp));
            auxTd(I) = SBLE_d.timestamp(I); % timestamps for n-th user           
            aux = trips(n).t{tripInd}.end - auxTd; aux(aux <= 0) = -inf;
            [~,ind2] = min(aux); % note that there may be more than 1 

            % test RM 12/1
            if ind2==1 
                disp('ind2 = 1 - need to look into it')
                ind2 = length(auxTd); 
            end

            II(ind1(1):ind2(end)) = I(ind1(1):ind2(end)); 

            % RM 11/30 Let's remove the last sample of a trip. This is
            % because sometimes it is the fist sample of the next
            % collecting_data, which could be far away.
            II(ind2(end)) = false;
            
            % RM 7/27/24 in case a trip has 0 samples, we shouldn't have a
            % pre-trip
            
            LLM(n).t{tripInd}.tM = [];
            LLM(n).t{tripInd}.tD1 = [];
            LLM(n).t{tripInd}.tD2 = [];
            % The following is probably useless since LL1 etc were not
            % defined yet
            LL1(n).t{tripInd} = []; LL2(n).t{tripInd} = []; LL(n).t{tripInd} = [];
            LL1pt(n).t{tripInd} = []; LL2pt(n).t{tripInd} = []; LLpt(n).t{tripInd} = [];
            theMM(n).t{tripInd} = 0; % this should actually be empty?
            allM(n).t{tripInd} = [];
            totS(n).t{tripInd} = [];
            nClips(n).t{tripInd} = [];
            totNS1or2_5(n).t{tripInd} = [];

             if (sum(II) == 0)
                IIpt = false(size(I));
                trips(n).t{1} = [];
            else

                % Find the logical indices of timestamps that are within the tripInd-th
                % pre-trip for user n

        %            aux = abs(SBLE_d.timestamp - pretrips(n).t{tripInd}.init);
                aux = SBLE_d.timestamp - pretrips(n).t{tripInd}.init; aux(aux < 0) = inf;
                [~,ind1] = min(aux);

        %           aux = abs(SBLE_d.timestamp - pretrips(n).t{tripInd}.end);
                aux = SBLE_d.timestamp - pretrips(n).t{tripInd}.end; aux(aux < 0) = inf;
                [~,ind2] = min(aux); %note that there may be more than 1 

                IIpt = false(size(I));
                % logical indices for user n for the pretrips of index tripInd
                IIpt(ind1(1):ind2(end)) = I(ind1(1):ind2(end));        

                % RM 12/29
                if (sum(II) > 0) && (sum(IIpt) > 0)
                    if ((min(SBLE_d.timestamp(II)) - max(SBLE_d.timestamp(IIpt)) > 300) || ...
                       (max(SBLE_d.timestamp(IIpt)) - min(SBLE_d.timestamp(IIpt)) > 300) || ...
                       (max(SBLE_d.timestamp(II)) - min(SBLE_d.timestamp(IIpt)) > 3000))
                        % What must have happened here is that the user clicked 'on
                        % the bus' at some point long after there was no more data.
                        % Or, never clicked
                        % The last test is for cases in which the trip is super
                        % long with a large gap. The beginning of the trip may
                        % be close to the pretrip. (Happens with n==28,
                        % tripInd==10)
                        % We should then discard pretrip.
                        IIpt = false(size(I));
                    end
                end

                % Let's put a check for the time distance between trip and
                % pretrip



        %            Set of majors seen during a pretrip + trip
                 theMajor = unique(SBLE_d.major(II | IIpt));

                % Set of majors seen during a trip
        %            theMajor = unique(SBLE_d.major(II));

                if ~isempty(theMajor)
                    time1 = zeros([1,length(theMajor)]); time2 = time1;
                    tdur = time1;

                    % Loop through all the majors seen
                    for m = 1:length(theMajor)

                        % Indices for m-th major 
                        IIm = SBLE_d.major == theMajor(m);  

                        maxLength = 0; 

                        % sequencial indices for n-th user
                        II_l = 1:length(I);

                        % Note: 
                        % II1&IIm&II = logical indices for m-th major, n-th user,
                        % tripInd-th trip, first minor

                        % total samples for  m-th major, n-th user,
                        % tripInd-th trip, first and second minor
                        time1(m) = sum((II1&IIm&II)) + sum((II1&IIm&IIpt));
                        time2(m) = sum((II2&IIm&II)) + sum((II2&IIm&IIpt));  

                        if ~isempty(II_l((II1|II2)&IIm&II))
                            % there was at least one minor recorded in this trip for
                            % this major

                            % total duration of recording at least one minor
                            tdur(m) = SBLE_d.timestamp(max(II_l((II1|II2)&IIm&II))) - ...
                                SBLE_d.timestamp(min(II_l((II1|II2)&IIm&II)));
                        end

                        LLM(n).t{tripInd}.tM(m) = theMajor(m);
                        LLM(n).t{tripInd}.tD1{m} = II_l(II1&IIm&II);
                        LLM(n).t{tripInd}.tD2{m} = II_l(II2&IIm&II);
                    end

                    % Check that this is not too long
                    if max(tdur) > 3600
                        % Discard
                        LLM(n).t{tripInd}.tM = [];
                        LLM(n).t{tripInd}.tD1 = [];
                        LLM(n).t{tripInd}.tD2 = [];
                        % The following is probably useless since LL1 etc were not
                        % defined yet
                        LL1(n).t{tripInd} = []; LL2(n).t{tripInd} = []; LL(n).t{tripInd} = [];
                        LL1pt(n).t{tripInd} = []; LL2pt(n).t{tripInd} = []; LLpt(n).t{tripInd} = [];
                    else
                        % Find the major with longest duration   
                        % ==> Is the right thing to do? time1 and time2 measure
                        % possibly overlapping periods of time. Maybe it would be
                        % better to measure the length of the union of times where
                        % either minor is received
                        [~,m_max] = max(time1+time2);

                        % Record data for this major
                        IIm_max = SBLE_d.major == theMajor(m_max);  

                       % Majority major
                        theMM(n).t{tripInd} = theMajor(m_max);

                       % Sequential indices for user n, trip tripInd, major m_max, and minor 1 or 2
                        LL1(n).t{tripInd} = II_l(II1&IIm_max&II); LL2(n).t{tripInd} = II_l(II2&IIm_max&II);  
                        % Juxtaposition of sequential indices for both minors
                        LL(n).t{tripInd} = [LL1(n).t{tripInd},LL2(n).t{tripInd}];
                        
                        % RM 7/27/24 it may be that the max major was
                        % measured only in pretrip but never in trip. In
                        % this case, let's cancel the trip
                        
                        if isempty(LL(n).t{tripInd})
                            LL1(n).t{tripInd} = []; LL2(n).t{tripInd} = []; LL1pt(n).t{tripInd} = []; LL2pt(n).t{tripInd} = [];
                        else

                            LL1pt(n).t{tripInd} = II_l(II1&IIm_max&IIpt); LL2pt(n).t{tripInd} = II_l(II2&IIm_max&IIpt); 
                            LLpt(n).t{tripInd} = [LL1pt(n).t{tripInd},LL2pt(n).t{tripInd}];

                           % total number of samples recorded for this trip
        %                    totS(n).t{tripInd} = sum(II);
                            % RM 7/25/24 - must consider *unique* timestamps!
                            totS(n).t{tripInd} = length(unique(SBLE_d.timestamp(II)));


                           % total number of samples observed for this trip from a major
                           % other than the majority
        %                    totSspur(n).t{tripInd} = sum(II) - sum(IIm_max&II);

                            totSspur(n).t{tripInd} = totS(n).t{tripInd} - length(unique(SBLE_d.timestamp(IIm_max&II)));

                           %  majors observed in this trip
                            allM(n).t{tripInd} = theMajor;
                        end
                    end
                else
                   disp('No majors seen - strange')
                end
            end

        end
        % OK now we need to clip together trip segments that belong to the
        % same trip. We will say that two consecutive pretrips+trip for the
        % same person are the same trip if (1) the first one had didNotMarkExit= true, (2)
        % they have the same majority
        % major, and (3) are less than 20 minutes from each other

        oldTrip = 1; nClips(n).t{1} = 0;
        totS(n).t{1} = 0; totSspur(n).t{1} = 0;
        for tripInd = 2:length(trips(n).t)
            nClips(n).t{tripInd} = 0;

             if ~isempty(trips(n).t{oldTrip}) & didNotMarkExit(n).t{oldTrip} & theMM(n).t{tripInd} == theMM(n).t{oldTrip} & (SBLE_d.timestamp(min(LLpt(n).t{tripInd})) - SBLE_d.timestamp(min(LL(n).t{oldTrip})) < 1200)
                 % ==> SHouldn't it be SBLE_d.timestamp(max(LL(n).t{oldTrip}) ?

%                disp([oldTrip,tripInd])
                 % We need to clip them together
                LL1(n).t{oldTrip} = [LL1(n).t{oldTrip}, LL1pt(n).t{tripInd}, LL1(n).t{tripInd}]; LL1(n).t{tripInd} = [];
                LL2(n).t{oldTrip} = [LL2(n).t{oldTrip}, LL2pt(n).t{tripInd}, LL2(n).t{tripInd}]; LL2(n).t{tripInd} = [];
                LL1pt(n).t{tripInd} = []; LL2pt(n).t{tripInd} = [];

                % LLM(n).t{oldTrip}.tM = union(LLM(n).t{oldTrip}.tM,LLM(n).t{tripInd}.tM);
                % RM 12/29
                allMajorsClip = union(LLM(n).t{oldTrip}.tM,LLM(n).t{tripInd}.tM);

                old_LLM(n) = LLM(n).t{oldTrip};
                for m = 1:length(allMajorsClip)
                    LLM(n).t{oldTrip}.tD1{m} = [];
                    LLM(n).t{oldTrip}.tD2{m} = [];

                    indM = find(old_LLM(n).tM == allMajorsClip(m));
                    if ~isempty(indM)
                        LLM(n).t{oldTrip}.tD1{m} = old_LLM(n).tD1{indM};
                        LLM(n).t{oldTrip}.tD2{m} = old_LLM(n).tD2{indM};
                    end

                    % now for the pretrip of indTrip. For now, we don't
                    % care about the major, though we should
                    LLM(n).t{oldTrip}.tD1{m} = [LLM(n).t{oldTrip}.tD1{m},LL1pt(n).t{tripInd}];
                    LLM(n).t{oldTrip}.tD2{m} = [LLM(n).t{oldTrip}.tD2{m},LL2pt(n).t{tripInd}];

                    indM = find(LLM(n).t{tripInd}.tM == allMajorsClip(m));
                    if ~isempty(indM)
                        LLM(n).t{oldTrip}.tD1{m} = [LLM(n).t{oldTrip}.tD1{m},LLM(n).t{tripInd}.tD1{indM}];
                        LLM(n).t{oldTrip}.tD2{m} = [LLM(n).t{oldTrip}.tD2{m},LLM(n).t{tripInd}.tD2{indM}];
                    end
                end  

                LLM(n).t{oldTrip}.tM = allMajorsClip;

    %                 LLM(n).t{oldTrip}.tD1 = [LLM(n).t{oldTrip}.tD1,LL1pt(n).t{tripInd},LLM(n).t{tripInd}.tD1];
    %                 LLM(n).t{oldTrip}.tD2 = [LLM(n).t{oldTrip}.tD2,LL2pt(n).t{tripInd},LLM(n).t{tripInd}.tD2];

                allM(n).t{oldTrip} = unique([allM(n).t{oldTrip};allM(n).t{tripInd}]); allM(n).t{tripInd} = [];

                totS(n).t{oldTrip} =  totS(n).t{oldTrip} +  totS(n).t{tripInd};  totS(n).t{tripInd} = 0;
                totSspur(n).t{oldTrip} =  totSspur(n).t{oldTrip} +  totSspur(n).t{tripInd};  totSspur(n).t{tripInd} = 0;
                theMM(n).t{tripInd} = 0;
                SL(n).t{tripInd} = [];

                % the previous segment had didNotMarkExit=true. Now let's
                % it inherit that of the new segment
                didNotMarkExit(n).t{oldTrip} = didNotMarkExit(n).t{tripInd};
                nClips(n).t{oldTrip} = nClips(n).t{oldTrip} + 1;
%                disp(['clip!'])
            else
                oldTrip = tripInd;
            end
        end

        % Now clean it up

        for tripInd = length(trips(n).t):-1:2
            if theMM(n).t{tripInd} == 0
                % remove
              LL1(n).t{tripInd} = []; LL1pt(n).t{tripInd} = [];  
              LL2(n).t{tripInd} = []; LL2pt(n).t{tripInd} = [];  
              LLM(n).t{tripInd} = [];
              allM(n).t{tripInd} = [];
              totS(n).t{tripInd} = [];
              totSspur(n).t{tripInd} = [];
              theMM(n).t{tripInd} = [];
              nClips(n).t{tripInd} = [];
              trips(n).t{tripInd} = [];
              pretrips(n).t{tripInd} = [];
              SL(n).t{tripInd} = [];
            end
        end


        for tripInd = 1:length(trips(n).t)
            % Let's compute durations now. It is important to only do it on
            % one major.

            t_pt = SBLE_d.timestamp([LL1pt(n).t{tripInd},LL2pt(n).t{tripInd}]);
            t_t = SBLE_d.timestamp([LL1(n).t{tripInd},LL2(n).t{tripInd}]);

            %  Total duration of trip + pretrip
            if ~isempty(t_pt) && ~isempty(t_t)
                tDur(n).t{tripInd} =  max(t_t) - min(t_pt);
            elseif ~isempty(t_t)
                tDur(n).t{tripInd} =  max(t_t) - min(t_t);
            else
                tDur(n).t{tripInd} = 0;
            end      
            

            % Total number of samples for trip + pretrip
            totNS(n).t{tripInd} = length(unique(SBLE_d.timestamp(...
                [LL1pt(n).t{tripInd},LL2pt(n).t{tripInd},LL1(n).t{tripInd},LL2(n).t{tripInd}])));
            
            % Now only for those majors with both minor
            if ~isempty(theMM(n).t{tripInd}) && theMM(n).t{tripInd} ~= 8 && theMM(n).t{tripInd} ~= 2 && theMM(n).t{tripInd} ~= 21
                totNS1or2(n).t{tripInd} = length(unique(SBLE_d.timestamp(...
                    [LL1pt(n).t{tripInd},LL2pt(n).t{tripInd},LL1(n).t{tripInd},LL2(n).t{tripInd}])));
                totNS1(n).t{tripInd} = length(unique(SBLE_d.timestamp(...
                    [LL1pt(n).t{tripInd},LL1(n).t{tripInd}])));
                totNS2(n).t{tripInd} = length(unique(SBLE_d.timestamp(...
                    [LL2pt(n).t{tripInd},LL2(n).t{tripInd}])));
                
                %RM 7/28/24
                II1or2 = false(size(I));
                II1or2([LL1pt(n).t{tripInd},LL2pt(n).t{tripInd},LL1(n).t{tripInd},LL2(n).t{tripInd}]) = true;
                II1or2_5 = II1or2 & Is5;
%                 LL1or2_5 = 1:length(I);
%                 LL1or2_5 = LL1or2(II1or2_5);
                totNS1or2_5(n).t{tripInd} = length(unique(SBLE_d.timestamp(II1or2_5)));
            else
                totNS1or2(n).t{tripInd} = 0;
                totNS1(n).t{tripInd} = 0;
                totNS2(n).t{tripInd} = 0;
                totNS_goodbus(n).t{tripInd} = 0;               
            end
            totNS1and2(n).t{tripInd} = totNS1(n).t{tripInd} + totNS2(n).t{tripInd} - totNS1or2(n).t{tripInd};
            
            % list of timestamps and RSSI for two minors for user n and trip
            % tripInd
            TSm1 = SBLE_d.timestamp(LL1(n).t{tripInd});
            TSm2 = SBLE_d.timestamp(LL2(n).t{tripInd});
            RSSIm1 = SBLE_d.rssi(LL1(n).t{tripInd});
            RSSIm2 = SBLE_d.rssi(LL2(n).t{tripInd});

            TSm12u = unique([TSm1(:);TSm2(:)]);

            % Now look at all timestamps for which both minors are seen
            RSSIdiff(n).t{tripInd} = [];
            RSSIm1_both(n).t{tripInd} = [];
            RSSIm2_both(n).t{tripInd} = [];

            isEmptyMinor(n).t{tripInd,1} = 0;
            isEmptyMinor(n).t{tripInd,2} = 0;
            aux1 = 0; aux2 = 0;

            % Couldn't I have just computed the intersetion of timestamps?
            for ts = 1:length(TSm12u)
                its1 = find(TSm1 == TSm12u(ts));
                its2 = find(TSm2 == TSm12u(ts));
                if ~isempty(its1) && ~isempty(its2)
                    % OK, we found a timestamp with both measurements. Note
                    % that there could be more than one! Odd but possible.
                    % Let's assume that they have all the same RSSI
                    RSSIdiff(n).t{tripInd} = [RSSIdiff(n).t{tripInd},RSSIm2(its2(1)) - RSSIm1(its1(1))];
                    RSSIm1_both(n).t{tripInd} = [ RSSIm1_both(n).t{tripInd}, RSSIm1(its1(1))];
                    RSSIm2_both(n).t{tripInd} = [ RSSIm2_both(n).t{tripInd}, RSSIm2(its2(1))];
                elseif isempty(its1)
                    aux1 =  aux1 + 1;
                elseif isempty(its2)
                    aux2 =  aux2 + 1;
                end
            end

            % for these majors, there is one minor missing (2 for 2, 8 and 1
            % for 21). So don't report missing minors for these

            lengthEitherMinor(n).t{tripInd} = length(TSm12u);

            if ~isempty(TSm12u)        
                if theMajor(m_max) ~= 21
                    isEmptyMinor(n).t{tripInd,1} = aux1;
                end
                if (theMajor(m_max) ~= 2) && (theMajor(m_max) ~= 8)                   
                    isEmptyMinor(n).t{tripInd,2} = aux2;
                end
            end

            theLat = SBLE_d.latitude(unique([LL1(n).t{tripInd},LL2(n).t{tripInd}]));
            theLong = SBLE_d.longitude(unique([LL1(n).t{tripInd},LL2(n).t{tripInd}]));
            theLatpt = SBLE_d.latitude(unique([LL1pt(n).t{tripInd},LL2pt(n).t{tripInd}]));
            theLongpt = SBLE_d.longitude(unique([LL1pt(n).t{tripInd},LL2pt(n).t{tripInd}]));

            % To measure length of routes, we need the function distance
            % (available since R2019b)
            if exist('distance')
                theLat = theLat(:); theLong = theLong(:); theLatpt = theLatpt(:); theLongpt = theLongpt(:);

                if length(theLat) > 1
                    N = length(theLat);
                    theL = sum(deg2km(distance([theLat(1:N-1),theLong(1:N-1)],[theLat(2:N),theLong(2:N)])));
                else
                    theL = 0;
                end
                if length(theLatpt) > 1
                    N = length(theLatpt);
                    theLpt = sum(deg2km(distance([theLatpt(1:N-1),theLongpt(1:N-1)],[theLatpt(2:N),theLongpt(2:N)])));
                else
                    theLpt = 0;
                end

                % theLpt and theL are the length (in Km) of pretrips and
                % trip

                if (theL + theLpt) > 10
                    disp('Something''s fishy - length > 10 Km - impossible')
                end

                % Cumulative data
                  totDist = [totDist,theL + theLpt];
                if (theL + theLpt) > 0
                    totTrips(n) = totTrips(n) + 1;
                end
            else
                theL = 0; theLpt = 0;
            end

            if DISPLAY_DATA   

                % RM 11/22/23 Note that here we are assuming that we are
                % only receiving data from the bus we are on. Need to
                % change that.
                % 
%                if DISPLAY_PLOT || (~isempty(nClips(n).t{tripInd}) &&  nClips(n).t{tripInd} > 0)
%                if DISPLAY_PLOT && ~isempty(nClips(n).t{tripInd})
                if DISPLAY_PLOT && ~isempty(trips(n).t{tripInd})

                    % Show plots of all routes. 

                    figure(1), clf
                    geoplot(theLat,theLong,'.','MarkerSize',14); hold on
                    geoplot(theLatpt,theLongpt,'.r','MarkerSize',14); 
                    % frame on whole campus
                    latlims = [36.97   37.005]; lonlims = [ -122.072 -122.047];
                    geolimits(latlims,lonlims)
                    h = gca; h.FontSize = 12;
    %                     disp([names{n},' - Trip #',int2str(tripInd),' ',trips(tripInd).seat_location,' Length (m) ',int2str(round((theL+theLpt)*1000)),...
    %                         ' pretrips: ', int2str(theTimes_pt), ' Trip: ', int2str(theTimes_t)])
                    if ~isempty(trips(n).t{tripInd}) % why?!?
                        disp([names{n},' n = ',int2str(n),' - Trip #',int2str(tripInd),' ',trips(n).t{tripInd}.seat_location,...
                            '#clips = ',int2str(nClips(n).t{tripInd}),...
                            ' Pretrip duration: ',int2str(pretrips(n).t{tripInd}.end - pretrips(n).t{tripInd}.init),...
                            ' Trip duration: ',int2str(trips(n).t{tripInd}.end - trips(n).t{tripInd}.init),...                        
                            ' Actual trip duration: ', int2str(round(tDur(n).t{tripInd})),' Seconds recorded: ',int2str(round(totNS(n).t{tripInd}))])
                    end
                    hold on

                    for st = 1:height(stopsGTFS)
                        geoplot(stopsGTFS.stop_lat(st),stopsGTFS.stop_lon(st),'^')
                    end
                    h=gca;
                    h.FontSize = 14;
                    h.LabelFontSizeMultiplier = 1.2;

                end

                if ~isempty([LL1,LL2,LL1pt,LL2pt]) && ~isempty(LLM(n).t{tripInd}) % RM 7/27/24 why?
                    % Show plots of received RSSI

                    Tmin = inf;
                            TS1 = SBLE_d.timestamp(LL1(n).t{tripInd}); 
                            if ~isempty(TS1)
                                Tmin = min([Tmin, TS1(1)]);
                            end
                            TS2 = SBLE_d.timestamp(LL2(n).t{tripInd}); 
                            if ~isempty(TS2)
                                Tmin = min([Tmin, TS2(1)]);
                            end

                            % Need to plot for both minors (beacons) using
                            % different colors
                            [~,is1,is2] = intersect(TS1,TS2);
                            [~,xs1,xs2] = setxor(TS1,TS2);

                            rssi_arr_aux = zeros(41);

                            ind_rss1 = []; ind_rss2 = [];                            
%                           if DISPLAY_PLOT || (~isempty(nClips(n).t{tripInd}) &&  nClips(n).t{tripInd}> 0)
%                           if DISPLAY_PLOT && ~isempty(nClips(n).t{tripInd}) 
                           if DISPLAY_PLOT && ~isempty(trips(n).t{tripInd}) 
                                figure(2),clf, hold on
                                % plot pretrips + trips
                                TS1pt = SBLE_d.timestamp((LL1pt(n).t{tripInd})); 
                                if ~isempty(TS1pt)
                                    Tmin = min([Tmin, TS1pt(1)]);
                                end
                                TS2pt = SBLE_d.timestamp((LL2pt(n).t{tripInd})); 
                                if ~isempty(TS2pt)
                                    Tmin = min([Tmin, TS2pt(1)]);
                                end
                                for m = 1:length(LLM(n).t{tripInd}.tM)

                                     TS1 = SBLE_d.timestamp(LLM(n).t{tripInd}.tD1{m}); 
                                    if ~isempty(TS1)
                                        Tmin = min([Tmin, TS1(1)]);
                                    end
                                    TS2 = SBLE_d.timestamp(LLM(n).t{tripInd}.tD2{m}); 
                                    if ~isempty(TS2)
                                        Tmin = min([Tmin, TS2(1)]);
                                    end
                                end
                                plotCols = [0 0.4470 0.7410;0.8500 0.3250 0.0980;0.4940 0.1840 0.556;0.4660 0.6740 0.1880;0.3010 0.7450 0.9330;0.6350 0.0780 0.1840;0.9290 0.6940 0.1250];
%                                plot(TS1pt-Tmin,SBLE_d.rssi(LL1pt(n).t{tripInd}),'xr',TS2pt-Tmin,SBLE_d.rssi(LL2pt(n).t{tripInd}),'or','MarkerSize',10), hold on
                                plot(TS1pt-Tmin,SBLE_d.rssi(LL1pt(n).t{tripInd}),'x','MarkerSize',10,'Color',plotCols(2,:)); 
                                plot(TS2pt-Tmin,SBLE_d.rssi(LL2pt(n).t{tripInd}),'o','MarkerSize',10,'Color',plotCols(2,:))
                                   % Need to plot for both minors (beacons) using
                                    % different colors
    %                                     [~,is1,is2] = intersect(TS1,TS2);
    %                                     [~,xs1,xs2] = setxor(TS1,TS2);
                                % find the dominant major
                                [~,theM_ind] = find(LLM(n).t{tripInd}.tM == theMM(n).t{tripInd});
                                TS1 = SBLE_d.timestamp(LLM(n).t{tripInd}.tD1{theM_ind}); TS2 = SBLE_d.timestamp(LLM(n).t{tripInd}.tD2{theM_ind}); 
%                                plot(TS1-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD1{theM_ind}),'x',TS2-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD2{theM_ind}),'o','MarkerSize',10)                              
                                plot(TS1-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD1{theM_ind}),'x','MarkerSize',10,'Color',plotCols(1,:));
                                plot(TS2-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD2{theM_ind}),'o','MarkerSize',10,'Color',plotCols(1,:));  
                                plotInd = 3;
                                for m = 1:length(LLM(n).t{tripInd}.tM)
                                    if m ~= theM_ind
                                        TS1 = SBLE_d.timestamp(LLM(n).t{tripInd}.tD1{m}); TS2 = SBLE_d.timestamp(LLM(n).t{tripInd}.tD2{m}); 
                      %                  plot(TS1-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD1{m}),'x',TS2-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD2{m}),'o','MarkerSize',10)
                                        plot(TS1-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD1{m}),'x','MarkerSize',10,'Color',plotCols(plotInd,:));
                                        plot(TS2-Tmin,SBLE_d.rssi(LLM(n).t{tripInd}.tD2{m}),'o','MarkerSize',10,'Color',plotCols(plotInd,:));
                                        plotInd = plotInd + 1;
                                    end
                                end
    %                                 plot(TS1-Tmin,SBLE_d.rssi(LL1(n).t{tripInd}),'xb',TS2-Tmin,SBLE_d.rssi(LL2(n).t{tripInd}),'ob','MarkerSize',10), hold on
                                grid
                                title(['Major: ',int2str(theMM(n).t{tripInd}), ' Exit marked: ',int2str(~didNotMarkExit(n).t{tripInd}),' Seat: ',int2str(SL(n).t{tripInd})])
                                ylim([-101 -65])
                                hold off
                                xlabel('Time (s)','FontSize',18)
                                ylabel('RSSI (dBm)','FontSize',18)
                                h=gca;
                                h.FontSize = 14;
                           end
                        end
    %                     end
    %                 end
%                if DISPLAY_PLOT || (~isempty(nClips(n).t{tripInd}) &&  nClips(n).t{tripInd}> 0)
                if DISPLAY_PLOT && ~isempty(trips(n).t{tripInd}) 
                    %pause(0.1)
                    pause
                end

    %            end
            end          
        end
    %   end

       % This measures the amount of time in which data was recorded outside of
       % a trip
        theTimesOut(n) = theTimesOut(n) - theTimes(n);
        theTimesOut5(n) = theTimesOut5(n) - theTimes5(n);
    end
end
%%
if DISPLAY_SEATS

for i = 1:3
    totDiff{i} = [];
    totNDiff{i} = [];
    totMDiff{i} = [];
    tot_RSSIm1_both{i} = [];
    tot_RSSIm2_both{i} = [];
    
end
 for n=1:length(trips)
    for tripInd = 1:length(trips(n).t)
        if ~isempty(SL(n).t{tripInd}) && length(RSSIdiff) >= n && ~isempty(RSSIdiff(n).t{tripInd})
            totDiff{SL(n).t{tripInd}} = [totDiff{SL(n).t{tripInd}},mean(RSSIdiff(n).t{tripInd})];
%            totDiff{SL(n).t{tripInd}} = [totDiff{SL(n).t{tripInd}},mean(RSSIdiff(n).t{tripInd}(RSSIdiff(n).t{tripInd} ~= 0))];
            tot_RSSIm1_both{SL(n).t{tripInd}} = [tot_RSSIm1_both{SL(n).t{tripInd}},mean(RSSIm1_both(n).t{tripInd})];
            tot_RSSIm2_both{SL(n).t{tripInd}} = [tot_RSSIm2_both{SL(n).t{tripInd}},mean(RSSIm2_both(n).t{tripInd})];
%             tot_RSSIm1_both{SL(n).t{tripInd}} = [tot_RSSIm1_both{SL(n).t{tripInd}},mean(RSSIm1_both(n).t{tripInd}(RSSIm1_both(n).t{tripInd} ~= 0))];
%             tot_RSSIm2_both{SL(n).t{tripInd}} = [tot_RSSIm2_both{SL(n).t{tripInd}},mean(RSSIm2_both(n).t{tripInd}(RSSIm2_both(n).t{tripInd} ~= 0))];
%             tot_RSSIm1_both{SL(n).t{tripInd}} = [tot_RSSIm1_both{SL(n).t{tripInd}},(RSSIm1_both(n).t{tripInd})];
%             tot_RSSIm2_both{SL(n).t{tripInd}} = [tot_RSSIm2_both{SL(n).t{tripInd}},(RSSIm2_both(n).t{tripInd})];
                
        end
    end
 end
 
figure(6), clf, hold on
scatter(tot_RSSIm1_both{2}+rand([1,length(tot_RSSIm1_both{2})])-0.5,tot_RSSIm2_both{2}+rand([1,length(tot_RSSIm1_both{2})])-0.5,500,...
     'MarkerFaceAlpha',0.5,'MarkerEdgeColor','none','MarkerFaceColor',[180 180 180]/255) %[247 176 0]/255)
%    'MarkerFaceAlpha',0.3,'MarkerEdgeColor','none','MarkerFaceColor',[100 136 30]/255)
scatter(tot_RSSIm1_both{1}+rand([1,length(tot_RSSIm1_both{1})])-0.5,tot_RSSIm2_both{1}+rand([1,length(tot_RSSIm1_both{1})])-0.5,500,...
     'MarkerFaceAlpha',0.5,'MarkerEdgeColor','none','MarkerFaceColor',[0 114 195]/255)
scatter(tot_RSSIm1_both{3}+rand([1,length(tot_RSSIm1_both{3})])-0.5,tot_RSSIm2_both{3}+rand([1,length(tot_RSSIm1_both{3})])-0.5,500,...
     'MarkerFaceAlpha',0.5,'MarkerEdgeColor','none','MarkerFaceColor',[234    71     0]/255)
% plot(tot_RSSIm1_both{2}+rand([2,length(tot_RSSIm1_both{2})])-0.5,tot_RSSIm2_both{2}+rand([1,length(tot_RSSIm1_both{2})])-0.5,...
%     'Color',[0.9 0.9 0.9],'Marker','.', 'MarkerSize',20,'LineStyle','none')
% plot(tot_RSSIm1_both{3}+rand([2,length(tot_RSSIm1_both{3})])-0.5,tot_RSSIm2_both{3}+rand([1,length(tot_RSSIm1_both{3})])-0.5,...
%     'Color',[0.5 0.5 0.5],'Marker','.', 'MarkerSize',20,'LineStyle','none')
 
axis square,grid
xlabel('Front beacon SRRI (dBm)','FontSize',22)
ylabel('Rear beacon SRRI (dBm)','FontSize',22)
h = gca; h.FontSize = 14;

 figure(5),clf
 for i = 1:3
     hold on
     plot(sort(totDiff{i}),(1:length(totDiff{i})) / length(totDiff{i}),'o-')
 end
 hold off
 
 Ith1 = -15:0; Ith2 = 0:15;
 
 minacc = -inf;
 for th1 = Ith1
     for th2 = Ith2
         th = [-inf th1 th2 inf];
         a = [];acc=0; 
         for i=1:3
             for j=1:3
                 a(i,j) = sum((totDiff{i} > th(j)) & totDiff{i} <= th(j+1)) / length(totDiff{i});
             end
             acc = acc+a(i,i)*length(totDiff{i})/(length(totDiff{1})+length(totDiff{2})+length(totDiff{3}));
         end
         if acc > minacc
             bth1 = th1; bth2 = th2;
             minacc = acc;
             ba = a;
         end
     end
 end
 
 ll = (length(totDiff{1})+length(totDiff{2})+length(totDiff{3}));
 
 disp(['Random accuracy: ',num2str((length(totDiff{1})/ll)^2 + (length(totDiff{2})/ll)^2 + (length(totDiff{3})/ll)^2)])
 disp(['Best accuracy: ', num2str(minacc)])
 disp(ba)
 disp([bth1 bth2])

    
end
%%
if exist('distance')

    disp(['Mean distance traversed: ',num2str(mean(totDist))])
    disp(['Max distance traversed: ',num2str(max(totDist))])
    disp(['Min distance traversed: ',num2str(min(totDist))])
end

totNS_tot = 0; tot_dr = 0;trips_tot = 0;
totNS1and2_tot = 0; totNS1or2_tot = 0; trips1or2_tot = 0;
totNS1or2_5_tot = 0; totNS_noclip_tot = 0; tot_dr_noclip = 0;
for n=1:length(totNS)
    for i=1:length(totNS(n).t)
        if totNS(n).t{i} > 0
            totNS_tot = totNS_tot + (totNS(n).t{i});
            trips_tot = trips_tot + 1;
            if totNS1or2(n).t{i} > 0
                trips1or2_tot = trips1or2_tot + 1;
            end
            tot_dr = tot_dr + tDur(n).t{i};
            totNS1or2_tot = totNS1or2_tot + totNS1or2(n).t{i};
            totNS1and2_tot = totNS1and2_tot + totNS1and2(n).t{i};
            if ~isempty(totNS1or2_5(n).t{i})
                totNS1or2_5_tot = totNS1or2_5_tot + totNS1or2_5(n).t{i};
            end
            if nClips(n).t{i} == 0
               totNS_noclip_tot = totNS_noclip_tot +  totNS(n).t{i};
               tot_dr_noclip = tot_dr_noclip + tDur(n).t{i};
            end
        end
    end
end
disp(['Total number of unique timestaps recorded: ', int2str(length(unique(SBLE_d.timestamp)))])
disp(['Total number of unique timestaps recorded at >5ms: ', int2str(length(unique(SBLE_d.timestamp(Is5))))])
disp(['Total number of unique timestaps in trips: ', int2str(totNS_tot)])
disp(['Total number of unique timestaps in trips at >5ms: ', int2str(sum(totNS1or2_5_tot))])
disp(['Total number of unique timestaps recorded in trips with no clips: ', int2str(totNS_noclip_tot)])
disp(['Total duration of trips with no clips: ', int2str(tot_dr_noclip),' - ratio: ',int2str(totNS_noclip_tot/tot_dr_noclip)])


totSspur_tot = 0;
for n=1:length(totSspur)
    for i=1:length(totSspur(n).t)
        if ~isempty(totSspur(n).t{i})
        totSspur_tot = totSspur_tot + (totSspur(n).t{i});
        end
    end
end
%disp(['Total number of spurious samples: ', int2str(totSspur_tot),' - ratio: ',num2str(totSspur_tot/totS_tot)])
disp(['Total trip duration: ', int2str(tot_dr),' - ratio: ',num2str(totNS_tot/tot_dr)])

disp(['Ratio of samples with both beacon vs at least one (in busses with both beacons): ',num2str(totNS1and2_tot/totNS1or2_tot)])
% trips_tot = 0;
% for n=1:length(trips)
%     for i=1:length(trips(n).t)
%         trips_tot = trips_tot + length(trips(n).t{i});
%     end
% end
% 
disp(['Total number of trips: ', int2str(trips_tot),' of which ',int2str(trips1or2_tot),' in busses with both beacons'])

% disp(['Poportion of trips with spurious beacon: ',num2str(totSpur / sum(NTrips4n))])
% disp(['Proportion of time in a trip with spurious beacon: ', num2str(sum(theTimesSpur)/sum(theTimesWithSpur))])
% disp(['Proportion of time in a trip with spurious beacon when at > 5m/s: ', num2str(sum(theTimesSpur5)/sum(theTimesWithSpur5))])
%%
% disp(['Proportion of time in a trip at > 5 m/s: ', num2str( sum(theTimes5)/sum(theTimes))])
% disp(['Proportion of time off trip at > 5 m/s: ', num2str(sum(theTimesOut5)/sum(theTimesOut))])
% disp(['Ratio of time off vs on trip: ',num2str(sum(theTimesOut)/sum(theTimes))])
% 
% % Remember that the bus with major 21 only had 1 beacon readable
% disp(['Number of trips with only 1 beacon (excluding major 21): ',int2str(nnz(whichM0~=21))])
% 
% disp(['Number of trips where at the end, collecting_data was not false: ',int2str(ndidNotMarkExit)])
% 





