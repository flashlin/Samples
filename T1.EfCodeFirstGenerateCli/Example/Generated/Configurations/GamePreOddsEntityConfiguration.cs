using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class GamePreOddsEntityConfiguration : IEntityTypeConfiguration<GamePreOddsEntity>
    {
        public void Configure(EntityTypeBuilder<GamePreOddsEntity> builder)
        {
            builder.ToTable("GamePreOdds");

            builder.HasKey(x => x.gamepreoddsid);

            builder.Property(x => x.gamepreoddsid)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.gameid)
                .HasColumnType("int(10,0)")
            ;

            builder.Property(x => x.leagueid)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.margint)
                .HasColumnType("")
                .IsRequired()
            ;

            builder.Property(x => x.otherstatus2)
                .HasColumnType("int(10,0)")
                .IsRequired()
            ;

            builder.Property(x => x.cs00)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs01)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs10)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs11)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs02)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs12)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs22)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs21)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs20)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs30)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs31)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs32)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs33)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs23)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs13)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs03)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs43)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs42)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs41)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs40)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs44)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs14)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs24)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs04)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs34)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs50)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs05)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs00prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs01prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs10prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs11prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs02prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs12prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs22prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs21prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs20prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs30prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs31prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs32prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs33prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs23prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs13prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs03prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs43prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs42prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs41prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs40prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs44prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs14prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs24prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs04prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs34prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs50prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.cs05prob)
                .HasColumnType("")
            ;

            builder.Property(x => x.jackpotprob)
                .HasColumnType("")
            ;

            builder.Property(x => x.status)
                .HasColumnType("bit")
                .IsRequired()
                .HasDefaultValue(true)
            ;

        }
    }
}
