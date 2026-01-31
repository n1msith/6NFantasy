# Just some notes on strategy
#
# MOTM players
# Bookie any time try scorer odds
# Captain MOTM or any time try scorer
#
# Fixture competitive_gap
#
"""
I'm playing Six Nations fantasy rugby. Please help select a team with these rules:

Team Selection:
- Can choose up to 15 players + one supersub (who gets 3x points)
- 3 back three, 2 centre, 1 fly-half, 1 scrum-half, 3 back row, 2 second-row, 2 prop and 1 hooker.
- Don't have to fill every position (can leave one or two positions empty)
- Maximum 4 players per country/club
- Supersub should typically be a bench player (not full 80 mins)
- Total fantasy_value cap of 230 points
- fantasy_value is how much a player costs
- points is how many points a player scored for a round
- Use as close to 230 as possible, should be at least >=225
- Consider fixture match ups (strength and weakness of each team)


- Create the team as a list
- Calculate total points and total fantasy value

The data is from the first 2 rounds (of 5 total rounds) in the file {your_filename.csv}.
"""

match_results = {'round1': 
            {'match1': {'france': 43, 'wales': 0},
             'match2': {'scotland': 31, 'italy': 19},
             'match3': {'ireland': 27, 'england': 22},
             },
            'round2': 
            {'match1': {'italy': 22, 'wales': 15},
             'match2': {'england': 26, 'france': 25},
             'match3': {'scotland': 18, 'ireland': 32},
            'round3': 
            {'match1': {'wales': 18, 'ireland': 27},
             'match2': {'england': 16, 'scotland': 15},
             'match3': {'italy': 24, 'france': 73},
             },
            'round4': 
            {'match1': {'ireland': 27, 'france': 42},
             'match2': {'scotland': 35, 'wales': 29},
             'match3': {'england': 47, 'italy': 24},
             },    
            'round5': 
            {'match1': {'ireland': 22, 'italy': 17},
             'match2': {'england': 68, 'wales': 14},
             'match3': {'france': 35, 'scotland': 16},
             },         
             }     
            }

"""            
Final adjusted team that's under budget:

G. Alldritt (France, Back Row) - 19.1
L. Bielle-Biarrey (France, Back Three) - 15.9
H. Jones (Scotland, Centre) - 17.9
L. Cannone (Italy, Back Row) - 14.4
T. Curry (England, Back Row) - 15.2
J. Morgan (Wales, Back Row) - 17.2
F. Smith (England, Fly-Half) - 9.4
C. Doris (Ireland, Back Row) - 17.9
R. Darge (Scotland, Back Row) - 15.3
D. Jenkins (Wales, Second Row) - 15.0
T. Attissogbe (France, Back Three) - 11.0
S. Prendergast (Ireland, Fly-Half) - 12.2
S. Negri (Italy, Back Row) - 13.0
C. Murley (England, Back Three) - 10.0
W. Rowlands (Wales, Second Row) - 10.9
Supersub: J. Conan (Ireland, Back Row) - 12.5

New total: 226.9 (under the 230 cap)
"""

