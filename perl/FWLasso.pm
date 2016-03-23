package FWLasso;
use strict;
use warnings;
use parent qw/Class::Accessor::Fast/;
use List::Util qw/shuffle/;
use constant {
    DEFAULT_BOUND  => 2,
    DEFAULT_MARGIN => 1,
    DEFAULT_WEIGHT => 0,
};

__PACKAGE__->mk_accessors(qw/
    bound
    margin
/);

sub new {
    my ($class) = @_;
    return $class->SUPER::new({
        bound  => DEFAULT_BOUND,
        margin => DEFAULT_MARGIN,
        weight => {},
    });
}

sub predict {
    my ($self, $data) = @_;

    my $inner_product = 0.0;
    foreach my $feature (keys %$data) {
        next unless ($self->{weight}{$feature});
        $inner_product += (
            $self->{weight}{$feature} * $data->{$feature}
        );
    }
    return $inner_product;
}

sub train {
    my ($self, $data, $label, $t) = @_;

    return if (($self->predict($data) * $label) > $self->margin);

    my $max_feature;
    foreach my $feature (shuffle keys %$data) {
        if ((not $max_feature) or (abs($data->{$feature}) > abs($data->{$max_feature}))) {
            $max_feature = $feature;
        }
    }
    my $v     = (($data->{$max_feature} * -1 * $label) > 0) ? -1 : 1;
    my $gamma = 2 / ($t + 2);

    $self->{weight}{$max_feature} = DEFAULT_WEIGHT unless (defined $self->{weight}{$max_feature});
    $self->{weight}{$max_feature} = (1 - $gamma) * $self->{weight}{$max_feature} + $gamma * $self->bound * $v;
}

1;
