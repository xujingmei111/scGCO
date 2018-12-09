open(IN, "middle_sample_info.csv") or die "dd";
while($line=<IN>){
    @arr=split /\,/, $line;
    $x{$arr[0]} = $arr[1];
    $y{$arr[0]} = $arr[2];   
}

open(IN, "middle_exp_mat.csv") or die "dd";
$line=<IN>;
$ll=$line;
$ll=~ s/\,/ /g;
print "$ll";
while($line=<IN>){
    @arr=split /\,/, $line;
    $coor = $x{$arr[0]}."x".$y{$arr[0]};
    print "$coor";
    for ($i = 1; $i<=$#arr; $i++){
	print " $arr[$i]";
    }
#    print "\n";
}
