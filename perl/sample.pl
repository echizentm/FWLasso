#!/usr/bin/perl
use strict;
use warnings;
use Data::Dumper;
use JSON::XS   qw/decode_json encode_json/;
use List::Util qw/shuffle/;
use FWLasso;

my $fwl = FWLasso->new();

my @objs;
while (my $line = <STDIN>) { push(@objs, decode_json($line)); }

foreach my $t (1 ... 1000) {
    foreach (shuffle @objs) {
        # Lassoによるスパース化の効果を確認するためにノイズ特徴量を入れてみる
        $_->{data}{noise} = (rand(1) > 0.5) ? 1 : 0;
        $fwl->train($_->{data}, $_->{label}, $t);
    }
}

print "weight: " . Dumper($fwl->{weight});
foreach (@objs) {
    print encode_json($_)."\n";
    print "predict: " . $fwl->predict($_->{data})."\n";
    print "\n";
}

