\cd database/phase1_dataset/

.moviedata.imdb_actor:1!("SSC"; enlist ",") 0:`$"imdb-actor-info.csv";
.movies.mlusers:("S ";enlist csv) 0: `$"mlusers.csv";
.moviedata.genome_tags:1!("SS"; enlist ",") 0:`$"genome-tags.csv";
.moviedata.mlmovies:1!("SSSS"; enlist ",") 0:`$"mlmovies.csv";
.moviedata.mltags:3!("SSSP"; enlist ",") 0:`$"mltags.csv";
.moviedata.movie_actor:2!("SSI"; enlist ",") 0:`$"movie-actor.csv";
.moviedata.mlratings:3!("SSSIP"; enlist ",") 0:`$"mlratings.csv";

// create foreign key constrains

update `.moviedata.mlmovies$movieid from  `.moviedata.movie_actor;
update `.moviedata.imdb_actor$actorid from  `.moviedata.movie_actor;
update `.moviedata.mlmovies$movieid from  `.moviedata.mlratings;
update `.moviedata.mlmovies$movieid from  `.moviedata.mltags;
update `.moviedata.genome_tags$tagid from  `.moviedata.mltags;
